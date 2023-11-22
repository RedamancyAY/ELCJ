
from importlib.resources import path
import os

from zmq import device


from toolkits.data import FrameFaceDataset
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist

from toolkits import utils, split
from tqdm import tqdm
import shutil
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torchsummary import summary
from config import config
import time

from model import models, xception
from torch.nn import functional as F
import torchstat as stat

###################
# distributed training
###################

# ! process marker
# * explaination marker
# ? testing code marker


store = torch.zeros((10))


def dist_gen(gpu, args):
    # !step 1
    # current gpu id
    rank = gpu
    print('Rank id: ', rank)

    # !step 2
    # turn args to variable
    train_datasets1 = args.traindb1
    train_datasets2 = args.traindb2
    teacher_path = args.net_t_path
    student_path = args.net_s_path
    trainIndex = args.trainIndex
    valIndex_1 = args.valIndex_1
    valIndex_2 = args.valIndex_2
    valIndex_3 = args.valIndex_3
    mode = args.mode
    ffpp_df_path = args.ffpp_faces_df_path
    ffpp_faces_dir = args.ffpp_faces_dir
    celeb_df_path = args.celebdf_faces_df_path
    celeb_faces_dir = args.celebdf_faces_dir
    face_policy = args.face
    face_size = args.size
    batch_size = args.batch
    initial_lr = args.lr
    validation_interval = args.valint
    patience = args.patience

    log_interval = args.logint
    num_workers = args.workers
    seed = args.seed
    debug = args.debug

    weights_folder = args.models_dir
    logs_folder = args.log_dir
    world_size = args.world_size
    backend = args.backend
    init_method = args.init_method
    epoch_run = args.epochs
    model_period = args.modelperiod
    tagnote = args.tagnote

    initial_model = args.index

    # initiate process group, decide the wat of process comunication
    dist.init_process_group(
        backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    torch.manual_seed(0)
    # ?model = ConvNet()

    # gain model class form variables
    model_s_class = getattr(models, args.net_s)

    model_s = model_s_class(teacher_path)

    # model_s.judge.j_linear.register_backward_hook(backward_hook1)
    # model_s = model_s_class()

    # total = sum([param.nelement() for param in model_s.parameters()])
    # print('  + Number of params: %.2fM' % (total / 1e6))
    # summary(model_s.cuda(), input_size=(3, 299, 299))

    transformer = utils.get_transformer(face_policy=face_policy, patch_size=face_size,
                                        net_normalizer=model_s.get_normalizer(), train=True)

    # generate tag
    tag = utils.make_train_tag(net_class=model_s_class,
                               traindb=train_datasets1,
                               face_policy=face_policy,
                               patch_size=face_size,
                               seed=seed,
                               debug=debug,
                               note=tagnote
                               )

    # generate paths list
    bestval_path = os.path.join(weights_folder, tag, 'bestval.pth')
    last_path = os.path.join(weights_folder, tag, 'last.pth')
    periodic_path = os.path.join(weights_folder, tag, 'it{:06d}.pth')
    path_list = [bestval_path, last_path, periodic_path.format(initial_model)]
    os.makedirs(os.path.join(weights_folder, tag), exist_ok=True)

    # !step3
    optimizer = torch.optim.Adam(
        model_s.get_trainable_parameters(), lr=initial_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=patience,
        cooldown=2 * patience,
        min_lr=initial_lr*1e-3,
    )

    # !step4
    val_loss = min_val_loss = 10
    epoch = iteration = 0

    # loading model
    load_model(model_s, optimizer, [student_path], 0,
               initial_model)
    # print(epoch)

    model_s = model_s.cuda(gpu)

    model_s = nn.parallel.DistributedDataParallel(
        model_s, device_ids=[gpu], find_unused_parameters=True)

    # transfer model to cuda
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(gpu)

    # !step5

    # setting the syncBN
    if args.syncbn:
        model_s = nn.SyncBatchNorm.convert_sync_batchnorm(model_s)
        if gpu == 0:
            print('Use SyncBN in training')
    torch.cuda.set_device(gpu)

    # !step6

    print("Loading data")

    # generate Dataframe for dataloading
    dfs_train1, dfs_val1 = split.make_splits_FFPP(
        ffpp_df_path, train_datasets1)
    dfs_train2, dfs_val2 = split.make_splits_celebdf(
        celeb_df_path, train_datasets2)
    dfs_train3, dfs_val3 = split.make_splits_dfdc(
        "/mnt/8T/hou/dfdc_faces/faces_df.pkl")

    train_dataset = FrameFaceDataset(root=ffpp_faces_dir,
                                     df=dfs_train1[trainIndex],
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )
    train_dataset1 = FrameFaceDataset(root=celeb_faces_dir,
                                      df=dfs_train2,
                                      scale=face_policy,
                                      transformer=transformer,
                                      size=face_size,
                                      )
    train_dataset2 = FrameFaceDataset(root="/mnt/8T/hou/dfdc_faces",
                                      df=dfs_train3,
                                      scale=face_policy,
                                      transformer=transformer,
                                      size=face_size,
                                      )
    val_dataset_1 = FrameFaceDataset(root=ffpp_faces_dir,
                                     df=dfs_val1[valIndex_1],
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )
    val_dataset_2 = FrameFaceDataset(root=ffpp_faces_dir,
                                     df=dfs_val1[valIndex_2],
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )
    val_dataset_3 = FrameFaceDataset(root=ffpp_faces_dir,
                                     df=dfs_val1[valIndex_3],
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )
    val_dataset_4 = FrameFaceDataset(root=celeb_faces_dir,
                                     df=dfs_val2,
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )
    val_dataset_5 = FrameFaceDataset(root="/mnt/8T/hou/dfdc_faces",
                                     df=dfs_val3,
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )

    if len(train_dataset) == 0:
        print('No training samples. Halt.')
        return

    print('Training samples: {}'.format(len(train_dataset)))
    print('Validation_1 samples: {}'.format(len(val_dataset_1)))
    print('Validation_2 samples: {}'.format(len(val_dataset_2)))
    print('Validation_3 samples: {}'.format(len(val_dataset_3)))
    print('Validation_3 samples: {}'.format(len(val_dataset_4)))

    # initiate data sampler for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    train_sampler1 = torch.utils.data.distributed.DistributedSampler(train_dataset1,
                                                                     num_replicas=world_size,
                                                                     rank=rank)
    train_sampler2 = torch.utils.data.distributed.DistributedSampler(train_dataset2,
                                                                     num_replicas=world_size,
                                                                     rank=rank)
    val_sampler_1 = torch.utils.data.distributed.DistributedSampler(val_dataset_1,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    val_sampler_2 = torch.utils.data.distributed.DistributedSampler(val_dataset_2,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    val_sampler_3 = torch.utils.data.distributed.DistributedSampler(val_dataset_3,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    val_sampler_4 = torch.utils.data.distributed.DistributedSampler(val_dataset_4,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    val_sampler_5 = torch.utils.data.distributed.DistributedSampler(val_dataset_5,
                                                                    num_replicas=world_size,
                                                                    rank=rank)

    # generate dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                sampler=train_sampler1)
    train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                sampler=train_sampler2)

    val_loader_1 = torch.utils.data.DataLoader(dataset=val_dataset_1,
                                               num_workers=num_workers,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=val_sampler_1)

    val_loader_2 = torch.utils.data.DataLoader(dataset=val_dataset_2,
                                               num_workers=num_workers,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=val_sampler_2)
    val_loader_3 = torch.utils.data.DataLoader(dataset=val_dataset_3,
                                               num_workers=num_workers,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=val_sampler_3)
    val_loader_4 = torch.utils.data.DataLoader(dataset=val_dataset_4,
                                               num_workers=num_workers,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=val_sampler_4)
    val_loader_5 = torch.utils.data.DataLoader(dataset=val_dataset_5,
                                               num_workers=num_workers,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=val_sampler_5)

    # !step 7
    logdir = os.path.join(logs_folder, tag)
    if iteration == 0:
        # delete the log if it already exist
        shutil.rmtree(logdir, ignore_errors=True)
    tb = SummaryWriter(logdir=logdir)

    # !step 8

    while epoch != epoch_run:
        # ?optimizer.zero_grad()

        train_loss = train_num = 0
        train_pred_list = []
        train_labels_list = []
        current = time.time()
        train_batch_loss = 0
        for train_batch in tqdm(train_loader2, desc='Epoch {:03d} '.format(epoch), leave=False,
                                total=len(train_loader2)):

            if (iteration % validation_interval == 0):

                # *Validation

                # val_loss = validation_routine_1(
                #     model_s, val_loader_1, val_loader_2, val_loader_3, val_loader_4, tb, iteration, 'val')

                val_loss = validation_routine_2(
                    model_s, val_loader_1, val_loader_5, tb, iteration, 'val')

                tb.flush()

                # adjust learning rate according to validation loss
                lr_scheduler.step(val_loss)

                # Model checkpoint
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_model_v2(model_s, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, bestval_path)
                save_model_v2(model_s, optimizer, train_loss, val_loss,
                              iteration, batch_size, epoch, periodic_path.format(iteration))

            model_s.train()

            (batch_data, batch_labels), batch_paths = train_batch
            train_batch_num = len(batch_labels)
            # calculate training amount
            train_num += train_batch_num

            start = time.time()
            # print(start-current)

            train_batch_loss, train_batch_pred = batch_forward(
                model_s=model_s,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths,
                flag=True)

            _, predicted = torch.max(train_batch_pred, 1)
            train_pred_list.append(predicted.cpu().numpy())
            train_labels_list.append(batch_labels.numpy().flatten())

            if torch.isnan(train_batch_loss):
                raise ValueError('NaN loss')

            train_loss += train_batch_loss.item() * train_batch_num

            # loss backward
            train_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # recording training information every log_interval
            if iteration > 0 and (iteration % log_interval == 0):
                train_loss /= train_num
                train_labels = np.concatenate(train_labels_list)
                train_pred = np.concatenate(train_pred_list)

                train_acc = accuracy_score(train_pred, train_labels)
                train_f1 = f1_score(train_pred, train_labels)
                train_roc_auc = roc_auc_score(train_labels, train_pred)

                tb.add_scalar('train/loss', train_loss, iteration)
                tb.add_scalar(
                    'train/lr', optimizer.param_groups[0]['lr'], iteration)
                tb.add_scalar('epoch', epoch, iteration)

                tb.add_scalar('train/acc', train_acc, iteration)
                tb.add_scalar('train/f1', train_f1, iteration)
                tb.add_scalar('train/roc_auc', train_roc_auc, iteration)
                tb.flush()

                # save model every model_period
                if (iteration % model_period == 0):

                    # save_model_v2(model_s, optimizer, train_loss, val_loss,
                    #               iteration, batch_size, epoch, periodic_path.format(iteration))
                    save_model_v2(model_s, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, last_path)

                train_loss = train_num = 0
                train_labels_list = []
                train_pred_list = []

            iteration = iteration + 1

            # current = time.time()
        # save_model_v2(model_s, optimizer, train_loss, val_loss,
        #               iteration, batch_size, epoch, periodic_path.format(epoch))
        epoch = epoch + 1


'''
method definition:
------------------
:param net          training model
:param device       training device
:param criterion    training loss function
:param data         training data (Batch_size,3,299,299)
:param labels       labels corresponding to data
-----------------
:return loss[float]          training loss   
:return pred[ndarray]        training prediction
'''


def batch_forward(model_s: nn.Module, data: torch.Tensor, labels: torch.Tensor, iteration, tb, paths, flag=False, taecher_maxmean=None):
    data = data.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    pred_s, w = model_s(data)

    if flag:
        loss = loss_cal(pred_s, w, labels, iteration, tb,)
    else:
        loss = nn.CrossEntropyLoss()(pred_s, labels.long().flatten())
    return loss, pred_s


def featureAttentionLossCal(w):

    w_s = nn.Softmax(dim=1)(w*100)
    w_T = w_s.transpose(0, 1)
    w_2 = torch.mm(w_s, w_T)
    w_2diag = torch.diag(w_2)

    ans = max((sum(sum(w_2)) - sum(w_2diag))-(1e-6), 0)
    # ans = (sum(sum(w_2)) - sum(w_2diag))

    # if(ans > 10):
    #     print(ans)

    return ans


def loss_cal(pred_s, w, labels, iteration, tb):
    loss = 0
    w_mean = w.mean(dim=0)
    loss1 = nn.CrossEntropyLoss()(pred_s, labels.long().flatten())
    loss3 = max(abs(w_mean[0:2048].mean() -
                (w_mean[2048:].mean())-0.00001), 0)
    loss4 = featureAttentionLossCal(w)
    loss3 = 0
    loss4 = 0
    while loss3 > (loss1/4):
        loss3 = loss3/2
    while loss4 > (loss1/4):
        loss4 = loss4/2
    loss = loss1+loss3+loss4
    if iteration % 50 == 0:
        tb.add_scalar('train/loss1', loss1, iteration)
        tb.add_scalar('train/loss3', loss3, iteration)
        tb.add_scalar('train/loss4', loss4, iteration)
        # tb.add_scalar('train/loss', loss, iteration)
    return loss


'''
:method definition :  distributed data parallel model loading 
---------------------
:param model        loading mdoel instance
:param optimizer    loading optimizer instance
:param path_list    model path list
:param mode         model choice
:param index        parameter for path
'''


def load_model(model: nn.Module, optimizer: torch.optim.Optimizer, path_list: str, mode: int, index: int, flag_t=False):
    if not os.path.exists(path_list[mode]):
        return 0, 0

    if flag_t:
        print("loading teacher model")
        whole = torch.load(path_list[mode])
        incomp_keys = model.load_state_dict(
            {k.replace('module.', ''): v for k, v in whole['model'].items()})
        print(incomp_keys)
        return
    print("loading student model")

    whole = torch.load(path_list[mode])
    # loading model parameter
    incomp_keys = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in whole['model'].items()})
    print(incomp_keys)
    # loading optimizer parameter
    # opt_state = whole['opt']
    # optimizer.load_state_dict(opt_state)

    # loading other parameter
    epoch = whole['epoch']
    iteration = whole['iteration']

    return epoch, iteration


'''
method definition : saving model weight
------------------
:param net[nn.Module]   model need to be saved 
:param optimizer    optimizer need to be saved 
:param train_loss   current training loss    
:param val_loss     current validation loss  
:param iteration    current iteration 
:param batch_size   the batch size used
:param epoch        current epoch 
:param path         model saving path
------------------
'''


def save_model_v2(model: nn.Module, optimizer: torch.optim.Optimizer,
                  train_loss: float, val_loss: float,
                  iteration: int, batch_size: int, epoch: int,
                  path: str):
    path = str(path)
    model_state_dict = model.state_dict()
    # optimizer_state_dict =optimizer.state_dict()
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].cpu()

    # for key in optimizer.:
    #     optimizer_state_dict[key] = optimizer_state_dict[key].cpu()
    state = dict(model=model_state_dict,
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 iteration=iteration,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)


'''
method definition:
-----------------
:param  net              model need to be validated  
:param  device           device to run 
:param  val_loader       data loader for validation
:param  criterion        loss function 
:param  tb               tensorboard instance 
:param  iteration        current iteration
:param  tag              val tag for tensorboard
----------------
:return val_loss         validation loss

'''


def validation_routine(net, val_loader_1, val_loader_2, tb, iteration, tag: str):
    # switch to eval mode
    net.eval()

    val_num_1 = 0
    val_loss_1 = 0.
    val_num_2 = 0
    val_loss_2 = 0.
    pred_list = list()
    labels_list = list()

    for val_data in tqdm(val_loader_1, desc='Validation', leave=False, total=len(val_loader_1)):

        (batch_data, batch_labels), batch_paths = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(
                model_s=net,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths
            )
        _, predicted = torch.max(val_batch_pred, 1)
        pred_list.append(predicted.cpu().numpy())
        val_num_1 += val_batch_num
        val_loss_1 += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss_1 /= val_num_1
    tb.add_scalar('{}/loss_1'.format(tag), val_loss_1, iteration)

    val_labels = np.concatenate(labels_list)
    val_pred = np.concatenate(pred_list)

    val_labels_t = val_labels.astype(int)
    acc_1 = accuracy_score(val_pred, val_labels_t)
    val_f1_1 = f1_score(val_pred, val_labels_t)
    tb.add_scalar('{}/acc_1'.format(tag), acc_1, iteration)
    tb.add_scalar('{}/f1_1'.format(tag), val_f1_1, iteration)
    # tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    pred_list = list()
    labels_list = list()

    for val_data in tqdm(val_loader_2, desc='Validation', leave=False, total=len(val_loader_2)):

        (batch_data, batch_labels), batch_paths = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(
                model_s=net,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths
            )
        _, predicted = torch.max(val_batch_pred, 1)
        pred_list.append(predicted.cpu().numpy())
        val_num_2 += val_batch_num
        val_loss_2 += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss_2 /= val_num_2
    tb.add_scalar('{}/loss_2'.format(tag), val_loss_2, iteration)

    val_labels = np.concatenate(labels_list)
    val_pred = np.concatenate(pred_list)

    val_labels_t = val_labels.astype(int)
    acc_2 = accuracy_score(val_pred, val_labels_t)
    val_f1_2 = f1_score(val_pred, val_labels_t)
    tb.add_scalar('{}/acc_2'.format(tag), acc_2, iteration)
    tb.add_scalar('{}/f1_2'.format(tag), val_f1_2, iteration)

    tb.add_scalar('{}/loss_avg'.format(tag),
                  (val_loss_1+val_loss_2)/2, iteration)
    tb.add_scalar('{}/acc_avg'.format(tag), (acc_1+acc_2)/2, iteration)

    return (val_loss_1+val_loss_2)/2


def validation_routine_1(net, val_loader_1, val_loader_2, val_loader_3, val_loader_4, tb, iteration, tag: str):
    # switch to eval mode
    net.eval()

    val_num_1 = 0
    val_loss_1 = 0.
    val_num_2 = 0
    val_loss_2 = 0.
    val_num_3 = 0
    val_loss_3 = 0.
    val_num_4 = 0
    val_loss_4 = 0.
    pred_list = list()
    labels_list = list()

    for val_data in tqdm(val_loader_1, desc='Validation1', leave=False, total=len(val_loader_1)):

        (batch_data, batch_labels), batch_paths = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(
                model_s=net,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths
            )
        _, predicted = torch.max(val_batch_pred, 1)
        pred_list.append(predicted.cpu().numpy())
        val_num_1 += val_batch_num
        val_loss_1 += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss_1 /= val_num_1
    tb.add_scalar('{}/loss_1'.format(tag), val_loss_1, iteration)

    val_labels = np.concatenate(labels_list)
    val_pred = np.concatenate(pred_list)

    val_labels_t = val_labels.astype(int)
    acc_1 = accuracy_score(val_pred, val_labels_t)
    val_f1_1 = f1_score(val_pred, val_labels_t)
    tb.add_scalar('{}/acc_1'.format(tag), acc_1, iteration)
    tb.add_scalar('{}/f1_1'.format(tag), val_f1_1, iteration)
    # tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    pred_list = list()
    labels_list = list()

    for val_data in tqdm(val_loader_2, desc='Validation2', leave=False, total=len(val_loader_2)):

        (batch_data, batch_labels), batch_paths = val_data
        # 给定batch大小
        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(
                model_s=net,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths
            )
        _, predicted = torch.max(val_batch_pred, 1)
        pred_list.append(predicted.cpu().numpy())
        val_num_2 += val_batch_num
        val_loss_2 += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss_2 /= val_num_2
    tb.add_scalar('{}/loss_2'.format(tag), val_loss_2, iteration)

    val_labels = np.concatenate(labels_list)
    val_pred = np.concatenate(pred_list)

    val_labels_t = val_labels.astype(int)
    acc_2 = accuracy_score(val_pred, val_labels_t)
    val_f1_2 = f1_score(val_pred, val_labels_t)
    tb.add_scalar('{}/acc_2'.format(tag), acc_2, iteration)
    tb.add_scalar('{}/f1_2'.format(tag), val_f1_2, iteration)

    pred_list = list()
    labels_list = list()

    for val_data in tqdm(val_loader_3, desc='Validation3', leave=False, total=len(val_loader_3)):

        (batch_data, batch_labels), batch_paths = val_data
        # 给定batch大小
        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(
                model_s=net,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths
            )
        _, predicted = torch.max(val_batch_pred, 1)
        pred_list.append(predicted.cpu().numpy())
        val_num_3 += val_batch_num
        val_loss_3 += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss_3 /= val_num_3
    tb.add_scalar('{}/loss_3'.format(tag), val_loss_3, iteration)

    val_labels = np.concatenate(labels_list)
    val_pred = np.concatenate(pred_list)

    val_labels_t = val_labels.astype(int)
    acc_3 = accuracy_score(val_pred, val_labels_t)
    val_f1_3 = f1_score(val_pred, val_labels_t)
    tb.add_scalar('{}/acc_3'.format(tag), acc_3, iteration)
    tb.add_scalar('{}/f1_3'.format(tag), val_f1_3, iteration)

    pred_list = list()
    labels_list = list()

    for val_data in tqdm(val_loader_4, desc='Validation4', leave=False, total=len(val_loader_4)):

        (batch_data, batch_labels), batch_paths = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(
                model_s=net,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths
            )
        _, predicted = torch.max(val_batch_pred, 1)
        pred_list.append(predicted.cpu().numpy())
        val_num_4 += val_batch_num
        val_loss_4 += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss_4 /= val_num_4
    tb.add_scalar('{}/loss_4'.format(tag), val_loss_4, iteration)

    val_labels = np.concatenate(labels_list)
    val_pred = np.concatenate(pred_list)

    val_labels_t = val_labels.astype(int)
    acc_4 = accuracy_score(val_pred, val_labels_t)
    val_f1_4 = f1_score(val_pred, val_labels_t)
    tb.add_scalar('{}/acc_4'.format(tag), acc_4, iteration)
    tb.add_scalar('{}/f1_4'.format(tag), val_f1_4, iteration)

    tb.add_scalar('{}/loss_avg'.format(tag),
                  (val_loss_1+val_loss_2+val_loss_3+val_loss_4)/4, iteration)
    tb.add_scalar('{}/acc_avg'.format(tag),
                  (acc_1+acc_2+acc_3+acc_4)/4, iteration)

    return val_loss_2


def validation_routine_2(net, val_loader_1, val_loader_2, tb, iteration, tag: str):
    # switch to eval mode
    net.eval()

    val_num_1 = 0
    val_loss_1 = 0.
    val_num_2 = 0
    val_loss_2 = 0.
    pred_list = list()
    labels_list = list()

    for val_data in tqdm(val_loader_1, desc='Validation1', leave=False, total=len(val_loader_1)):

        (batch_data, batch_labels), batch_paths = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(
                model_s=net,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths
            )
        _, predicted = torch.max(val_batch_pred, 1)
        pred_list.append(predicted.cpu().numpy())
        val_num_1 += val_batch_num
        val_loss_1 += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss_1 /= val_num_1
    tb.add_scalar('{}/loss_1'.format(tag), val_loss_1, iteration)

    val_labels = np.concatenate(labels_list)
    val_pred = np.concatenate(pred_list)

    val_labels_t = val_labels.astype(int)
    acc_1 = accuracy_score(val_pred, val_labels_t)
    val_f1_1 = f1_score(val_pred, val_labels_t)
    tb.add_scalar('{}/acc_1'.format(tag), acc_1, iteration)
    tb.add_scalar('{}/f1_1'.format(tag), val_f1_1, iteration)
    # tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    pred_list = list()
    labels_list = list()

    for val_data in tqdm(val_loader_2, desc='Validation2', leave=False, total=len(val_loader_2)):

        (batch_data, batch_labels), batch_paths = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(
                model_s=net,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths
            )
        _, predicted = torch.max(val_batch_pred, 1)
        pred_list.append(predicted.cpu().numpy())
        val_num_2 += val_batch_num
        val_loss_2 += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss_2 /= val_num_2
    tb.add_scalar('{}/loss_2'.format(tag), val_loss_2, iteration)

    val_labels = np.concatenate(labels_list)
    val_pred = np.concatenate(pred_list)

    val_labels_t = val_labels.astype(int)
    acc_2 = accuracy_score(val_pred, val_labels_t)
    val_f1_2 = f1_score(val_pred, val_labels_t)
    tb.add_scalar('{}/acc_2'.format(tag), acc_2, iteration)
    tb.add_scalar('{}/f1_2'.format(tag), val_f1_2, iteration)

    tb.add_scalar('{}/loss_avg'.format(tag),
                  (val_loss_1+val_loss_2)/2, iteration)
    tb.add_scalar('{}/acc_avg'.format(tag),
                  (acc_1+acc_2)/2, iteration)

    return val_loss_2


def backward_hook1(module, grad_in, grad_out):
    # print(grad_in)
    # print(grad_out)
    grad_out = torch.mean(abs(grad_out[0]), dim=0).cpu()
    updateStore(grad_out)


def updateStore(add):
    global store
    store = store + add


def main():

    args = config.config_gen()
    # mp.spawn(dist_singlegen,  nprocs=args.gpus, args=(args,))

    mp.spawn(dist_gen,  nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()
