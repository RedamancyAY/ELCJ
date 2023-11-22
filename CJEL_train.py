
from importlib.resources import path
import os


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

from config import config
import time

from model import models, xception
from torch.nn import functional as F

###################
# distributed training
###################

# ! process marker
# * explaination marker
# ? testing code marker


def dist_EL(gpu, args):
    # !step 1
    # current gpu id
    rank = gpu
    print('Rank id: ', rank)

    # !step 2
    # turn args to variable

    train_datasets = args.traindb
    teacher_path = args.net_t_path
    trainIndex = args.trainIndex
    mode = args.mode
    ffpp_df_path = args.ffpp_faces_df_path
    ffpp_faces_dir = args.ffpp_faces_dir
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
    model_t_class = getattr(xception, args.net_t)

    model_s = model_s_class(teacher_path)
    model_t = model_t_class()

    transformer = utils.get_transformer(face_policy=face_policy, patch_size=face_size,
                                        net_normalizer=model_s.get_normalizer(), train=True)

    # generate tag
    tag = utils.make_train_tag(net_class=model_s_class,
                               traindb=train_datasets,
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
        params=[{'params': model_s.get_b1_trainable_parameters(), 'lr': initial_lr},
                {'params': model_s.get_b2_trainable_parameters(), 'lr': initial_lr*10},
                {'params': model_s.get_fp_trainable_parameters(), 'lr': initial_lr},
                {'params': model_s.get_j_trainable_parameters(), 'lr': initial_lr}
                ])
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
    epoch, iteration = load_model(model_s, optimizer, path_list, mode,
                                  initial_model)
    load_model(model_t, optimizer, [teacher_path],
               0, initial_model, flag_t=True)
    print(epoch)

    model_s = model_s.cuda(gpu)
    model_t = model_t.cuda(gpu)

    model_s = nn.parallel.DistributedDataParallel(
        model_s, device_ids=[gpu], find_unused_parameters=True)
    model_t = nn.parallel.DistributedDataParallel(
        model_t, device_ids=[gpu], find_unused_parameters=True)

    # transfer optimizer to cuda
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

    # *loading data
    print("Loading data")

    # generate Dataframe for dataloading
    dfs_train, dfs_val = split.make_splits_FFPP(ffpp_df_path, train_datasets)

    train_dataset = FrameFaceDataset(root=ffpp_faces_dir,
                                     df=dfs_train[trainIndex],
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )
    val_dataset_1 = FrameFaceDataset(root=ffpp_faces_dir,
                                     df=dfs_val[trainIndex],
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )

    if len(train_dataset) == 0:
        print('No training samples. Halt.')
        return

    print('Training samples: {}'.format(len(train_dataset)))
    print('Validation samples: {}'.format(len(val_dataset_1)))

    # initiate data sampler for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_1,
                                                                  num_replicas=world_size,
                                                                  rank=rank)

    # generate dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset_1,
                                             num_workers=num_workers,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             sampler=val_sampler)

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
        for train_batch in tqdm(train_loader, desc='Epoch {:03d} '.format(epoch), leave=False,
                                total=len(train_loader)):

            model_s.train()
            model_t.eval()

            (batch_data, batch_labels), batch_paths = train_batch
            train_batch_num = len(batch_labels)

            train_num += train_batch_num

            start = time.time()
            # print(start-current)

            train_batch_loss, train_batch_pred = batch_forward_EL(
                model_s=model_s,
                model_t=model_t,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths,
                flag=True)
            # print(time.time()-start)

            _, predicted = torch.max(train_batch_pred, 1)
            # correct += (predicted == batch_labels).sum().item()
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

            # validate model every val_interval
            if iteration > 0 and (iteration % validation_interval == 0):

                # *Validation

                val_loss = validation_routine_EL(
                    model_s, val_loader, tb, iteration, 'val')
                tb.flush()

                # adjust learning rate according to val_loss
                lr_scheduler.step(val_loss)

                # Model checkpoint
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_model_v2(model_s, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, bestval_path)

            iteration = iteration + 1
            # current = time.time()
        epoch = epoch + 1
        save_model_v2(model_s, optimizer, train_loss, val_loss,
                      iteration, batch_size, epoch, last_path)


def dist_train(gpu, args):
    # !step 1
    rank = gpu
    print('Rank id: ', rank)

    # !step 2

    train_datasets = args.traindb
    teacher_path = args.net_t_path
    trainIndex = args.trainIndex
    mode = args.mode
    ffpp_df_path = args.ffpp_faces_df_path
    ffpp_faces_dir = args.ffpp_faces_dir
    face_policy = args.face
    face_size = args.size
    batch_size = args.batch_train
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
    model_t_class = getattr(xception, args.net_t)

    model_s = model_s_class(teacher_path)
    model_t = model_t_class()

    transformer = utils.get_transformer(face_policy=face_policy, patch_size=face_size,
                                        net_normalizer=model_s.get_normalizer(), train=True)

    # generate tag
    tag = utils.make_train_tag(net_class=model_s_class,
                               traindb=train_datasets,
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
        params=[{'params': model_s.get_b1_trainable_parameters(), 'lr': initial_lr},
                {'params': model_s.get_b2_trainable_parameters(), 'lr': initial_lr*10},
                {'params': model_s.get_fp_trainable_parameters(), 'lr': initial_lr},
                {'params': model_s.get_j_trainable_parameters(), 'lr': initial_lr}
                ])
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
    epoch, iteration = load_model(model_s, optimizer, path_list, mode,
                                  initial_model)
    load_model(model_t, optimizer, [teacher_path],
               0, initial_model, flag_t=True)
    print(epoch)

    model_s = model_s.cuda(gpu)
    model_t = model_t.cuda(gpu)

    model_s = nn.parallel.DistributedDataParallel(
        model_s, device_ids=[gpu], find_unused_parameters=True)
    model_t = nn.parallel.DistributedDataParallel(
        model_t, device_ids=[gpu], find_unused_parameters=True)

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
    dfs_train, dfs_val = split.make_splits_FFPP(ffpp_df_path, train_datasets)

    train_dataset = FrameFaceDataset(root=ffpp_faces_dir,
                                     df=dfs_train[trainIndex],
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )
    val_dataset_1 = FrameFaceDataset(root=ffpp_faces_dir,
                                     df=dfs_val[trainIndex],
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )

    if len(train_dataset) == 0:
        print('No training samples. Halt.')
        return

    print('Training samples: {}'.format(len(train_dataset)))
    print('Validation samples: {}'.format(len(val_dataset_1)))

    # initiate data sampler for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_1,
                                                                  num_replicas=world_size,
                                                                  rank=rank)

    # generate dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset_1,
                                             num_workers=num_workers,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             sampler=val_sampler)

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
        for train_batch in tqdm(train_loader, desc='Epoch {:03d} '.format(epoch), leave=False,
                                total=len(train_loader)):

            model_s.train()
            model_t.eval()

            (batch_data, batch_labels), batch_paths = train_batch
            train_batch_num = len(batch_labels)
            # calculate training amount
            train_num += train_batch_num

            start = time.time()
            # print(start-current)

            # train_batch_loss, train_batch_pred = batch_forward_train(
            #     model_s=model_s,
            #     data=batch_data,
            #     labels=batch_labels,
            #     iteration=iteration,
            #     tb=tb,
            #     paths=batch_paths,
            #     flag=True)

            train_batch_loss, train_batch_pred = batch_forward_trainBilin(
                model_s=model_s,
                model_t=model_t,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths,
                flag=True)
            # print(time.time()-start)

            _, predicted = torch.max(train_batch_pred, 1)
            # correct += (predicted == batch_labels).sum().item()
            train_pred_list.append(predicted.cpu().numpy())
            train_labels_list.append(batch_labels.numpy().flatten())

            if torch.isnan(train_batch_loss):
                raise ValueError('NaN loss')

            train_loss += train_batch_loss.item() * train_batch_num

            # *运用优化器
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

                if (iteration % model_period == 0):

                    # save_model_v2(model_s, optimizer, train_loss, val_loss,
                    #               iteration, batch_size, epoch, periodic_path.format(iteration))
                    save_model_v2(model_s, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, last_path)

                train_loss = train_num = 0
                train_labels_list = []
                train_pred_list = []

            # validate model
            if iteration > 0 and (iteration % validation_interval == 0):

                # *Validation

                val_loss = validation_routine_train(
                    model_s, val_loader, tb, iteration, 'val')
                tb.flush()

                lr_scheduler.step(val_loss)

                # Model checkpoint
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_model_v2(model_s, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, bestval_path)

            iteration = iteration + 1
            # current = time.time()
        epoch = epoch + 1


'''
method definition:
------------------
:param model_s      training model
:param data         training data (Batch_size,3,299,299)
:param labels       labels corresponding to data
-----------------
:return loss[float]          training loss   
:return pred[ndarray]        training prediction
'''


def batch_forward_EL(model_s: nn.Module, model_t: nn.Module, data: torch.Tensor, labels: torch.Tensor, iteration, tb, paths, flag=False, taecher_maxmean=None):
    data = data.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    pred_s, w = model_s(data)

    if flag:
        pred_t = model_t(data)
        loss = Loss_cal_EL(pred_s, pred_t, w, labels, iteration, tb)
    else:
        loss = nn.CrossEntropyLoss()(pred_s, labels.long().flatten())

    return loss, pred_s


def batch_forward_train(model_s: nn.Module, data: torch.Tensor, labels: torch.Tensor, iteration, tb, paths, flag=False, taecher_maxmean=None):
    data = data.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    pred_s, w = model_s(data)

    if flag:

        loss = Loss_cal_train(pred_s, w, labels, iteration, tb)
    else:
        loss = nn.CrossEntropyLoss()(pred_s, labels.long().flatten())

    return loss, pred_s


def batch_forward_trainBilin(model_s: nn.Module, model_t: nn.Module, data: torch.Tensor, labels: torch.Tensor, iteration, tb, paths, flag=False, taecher_maxmean=None):
    data = data.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    pred_s, w = model_s(data)
    pred_t = model_t(data)

    if flag:

        loss = Loss_cal_train(pred_s, w, labels, iteration, tb)
    else:
        loss = nn.CrossEntropyLoss()(pred_s, labels.long().flatten())

    return loss, pred_s


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
    opt_state = whole['opt']
    optimizer.load_state_dict(opt_state)

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


def validation_routine_EL(net, val_loader, tb, iteration, tag: str):
    # switch to eval mode
    net.eval()

    val_num = 0
    val_loss = 0.
    pred_list = list()
    labels_list = list()
    for val_data in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader)):

        (batch_data, batch_labels), batch_paths = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward_EL(
                model_s=net,
                model_t=None,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths
            )
        _, predicted = torch.max(val_batch_pred, 1)
        pred_list.append(predicted.cpu().numpy())
        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss /= val_num
    tb.add_scalar('{}/loss'.format(tag), val_loss, iteration)

    val_labels = np.concatenate(labels_list)
    val_pred = np.concatenate(pred_list)
    # val_roc_auc = roc_auc_score(val_labels, val_pred)
    # val_pred_t = np.array([1 if item > 0 else 0 for item in val_pred])
    val_labels_t = val_labels.astype(int)
    acc = accuracy_score(val_pred, val_labels_t)
    val_f1 = f1_score(val_pred, val_labels_t)
    tb.add_scalar('{}/acc'.format(tag), acc, iteration)
    #val_f1 = f1_score(val_labels, val_pred)
    # tb.add_scalar('{}/roc_auc'.format(tag), val_roc_auc, iteration)
    tb.add_scalar('{}/f1'.format(tag), val_f1, iteration)
    tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    return val_loss


def validation_routine_train(net, val_loader, tb, iteration, tag: str):
    # switch to eval mode
    net.eval()

    val_num = 0
    val_loss = 0.
    pred_list = list()
    labels_list = list()
    for val_data in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader)):

        (batch_data, batch_labels), batch_paths = val_data
        # 给定batch大小
        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward_train(
                model_s=net,
                data=batch_data,
                labels=batch_labels,
                iteration=iteration,
                tb=tb,
                paths=batch_paths
            )
        _, predicted = torch.max(val_batch_pred, 1)
        pred_list.append(predicted.cpu().numpy())
        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss /= val_num
    tb.add_scalar('{}/loss'.format(tag), val_loss, iteration)

    val_labels = np.concatenate(labels_list)
    val_pred = np.concatenate(pred_list)
    # val_roc_auc = roc_auc_score(val_labels, val_pred)
    # val_pred_t = np.array([1 if item > 0 else 0 for item in val_pred])
    val_labels_t = val_labels.astype(int)
    acc = accuracy_score(val_pred, val_labels_t)
    val_f1 = f1_score(val_pred, val_labels_t)
    tb.add_scalar('{}/acc'.format(tag), acc, iteration)
    #val_f1 = f1_score(val_labels, val_pred)
    # tb.add_scalar('{}/roc_auc'.format(tag), val_roc_auc, iteration)
    tb.add_scalar('{}/f1'.format(tag), val_f1, iteration)
    tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    return val_loss


def Loss_cal_EL(outputs_s, outputs_t, w, labels, iteration, tb):
    # loss 1 BCE
    loss1 = nn.CrossEntropyLoss()(outputs_s, labels.long().flatten())

    # loss 2 KDLoss
    # when acc(stu) is higher loss 2 is useless
    if iteration == 0:
        loss2 = 0
    else:
        # normal_KD
        KD_alpha = 0.5
        KD_T = 10
        loss2_1 = KD_alpha*KD_T*KD_T*nn.KLDivLoss(reduction='batchmean')((nn.LogSoftmax(
            dim=0)(outputs_s/KD_T)), nn.Softmax(dim=0)(outputs_t/KD_T))

        # variational_KD
        loss2_2 = variationKDLossCal(
            outputs_s, outputs_t, KD_T=10, KD_alpha=0.5)
        loss2 = (loss2_2).cuda()
        # loss2 = 0
    # loss 3 EnergyLoss

    w_mean = w.mean(dim=0)
    loss3 = abs(w_mean[0:2048].mean()-(w_mean[2048:].mean()))*100
    # # loss 4 featureAttentionLoss
    loss4 = featureAttentionLossCal(w)

    # loss2 = 0
    # loss3 = 0
    # loss4 = 0
    loss = 0

    while loss2 > (loss1/4):
        loss2 = loss2/2
    while loss3 > (loss1/4):
        loss3 = loss3/2
    while loss4 > (loss1/4):
        loss4 = loss4/2
    if iteration % 100 == 0:
        tb.add_scalar('train/loss1', loss1, iteration)
        tb.add_scalar('train/loss2', loss2, iteration)
        tb.add_scalar('train/loss3', loss3, iteration)
        tb.add_scalar('train/loss4', loss4, iteration)

    loss = loss1+loss2+loss3+loss4
    if loss > 10:
        print(loss)

    return loss


def Loss_cal_train(outputs_s, w, labels, iteration, tb):
    # loss 1 BCE
    loss1 = nn.CrossEntropyLoss()(outputs_s, labels.long().flatten())
    # loss 2 KDLoss

    # if iteration > 1000:
    #     loss2 = 0
    # else:
    #     # normal_KD
    #     KD_alpha = 0.5
    #     KD_T = 10
    #     loss2_1 = KD_alpha*KD_T*KD_T*nn.KLDivLoss(reduction='batchmean')((nn.LogSoftmax(
    #         dim=0)(outputs_s/KD_T)), nn.Softmax(dim=0)(outputs_t/KD_T))

    # variational_KD
    # loss2_2 = variationKDLossCal(
    #     outputs_s, outputs_t, KD_T=10, KD_alpha=0.5)
    # loss2 = (loss2_2).cuda()
    # loss 3 EnergyLoss

    w_mean = w.mean(dim=0)
    loss3 = max(abs(w_mean[0:2048].mean() -
                (w_mean[2048:].mean())-0.00001), 0)
    # # loss 4 featureAttentionLoss
    loss4 = featureAttentionLossCal(w)

    # loss2 = 0
    # loss3 = 0
    # loss4 = 0
    loss = 0
    loss2 = 0
    loss3 = 0
    while loss2 > (loss1/4):
        loss2 = loss2/2
    # loss2 = 0
    while loss3 > (loss1/4):
        loss3 = loss3/2
    while loss4 > (loss1/4):
        loss4 = loss4/2
    if iteration % 100 == 0:
        tb.add_scalar('train/loss1', loss1, iteration)
        tb.add_scalar('train/loss2', loss2, iteration)
        tb.add_scalar('train/loss3', loss3, iteration)
        tb.add_scalar('train/loss4', loss4, iteration)

    loss = loss1+loss2+loss3+loss4
    if loss > 10:
        print(loss)

    return loss


def Loss_cal_trainBilin(outputs_s, outputs_t, w, labels, iteration, tb):
    # loss 1 BCE
    loss1 = nn.CrossEntropyLoss()(outputs_s, labels.long().flatten())
    # loss 2 KDLoss

    # if iteration > 1000:
    #     loss2 = 0
    # else:
    #     # normal_KD
    #     KD_alpha = 0.5
    #     KD_T = 10
    #     loss2_1 = KD_alpha*KD_T*KD_T*nn.KLDivLoss(reduction='batchmean')((nn.LogSoftmax(
    #         dim=0)(outputs_s/KD_T)), nn.Softmax(dim=0)(outputs_t/KD_T))

    # variational_KD
    loss2_2 = variationKDLossCal(
        outputs_s, outputs_t, KD_T=10, KD_alpha=0.5)
    loss2 = (loss2_2).cuda()
    # loss 3 EnergyLoss

    w_mean = w.mean(dim=0)
    loss3 = max(abs(w_mean[0:2048].mean() -
                (w_mean[2048:].mean())-0.00001), 0)
    # # loss 4 featureAttentionLoss
    loss4 = featureAttentionLossCal(w)

    # loss2 = 0
    # loss3 = 0
    # loss4 = 0
    loss = 0
    loss2 = 0
    loss3 = 0
    while loss2 > (loss1/4):
        loss2 = loss2/2
    # loss2 = 0
    while loss3 > (loss1/4):
        loss3 = loss3/2
    while loss4 > (loss1/4):
        loss4 = loss4/2
    if iteration % 100 == 0:
        tb.add_scalar('train/loss1', loss1, iteration)
        tb.add_scalar('train/loss2', loss2, iteration)
        tb.add_scalar('train/loss3', loss3, iteration)
        tb.add_scalar('train/loss4', loss4, iteration)

    loss = loss1+loss2+loss3+loss4
    if loss > 10:
        print(loss)

    return loss


def featureAttentionLossCal(w):

    # w = w[3:]
    w_s = nn.Softmax(dim=1)(w*100)
    w_T = w_s.transpose(0, 1)
    w_2 = torch.mm(w_s, w_T)
    w_2diag = torch.diag(w_2)

    ans = max((sum(sum(w_2)) - sum(w_2diag))-(1e-6), 0)
    # ans = (sum(sum(w_2)) - sum(w_2diag))
    # if(ans > 10):
    #     print(ans)

    return ans


def variationKDLossCal(pred_s, pred_t, KD_T=10, KD_alpha=0.5):
    int_s = torch.zeros(10)
    int_t = torch.zeros(10)
    pred_t = torch.sigmoid(pred_t)
    for i in range(len(pred_s)):
        temp1 = pred2Intreval(pred_s[i][0])
        temp2 = pred2Intreval(pred_s[i][1])
        int_s[temp1] = int_s[temp1] + pred_s[i][0]-(temp1*0.2)
        int_s[temp2+5] = int_s[temp2+5] + pred_s[i][1]-(temp2*0.2)
    for i in range(len(pred_t)):
        temp1 = pred2Intreval(pred_t[i][0])
        temp2 = pred2Intreval(pred_t[i][1])
        int_t[temp1] = int_t[temp1] + pred_t[i][0]-(temp1*0.2)
        int_t[temp2+5] = int_t[temp2+5] + pred_t[i][1]-(temp2*0.2)
    loss = KD_alpha*KD_T*KD_T*nn.KLDivLoss(reduction='batchmean')((nn.LogSoftmax(
        dim=0)(int_s/KD_T)), nn.Softmax(dim=0)(int_t/KD_T))
    return loss


def pred2Intreval(pred):
    interval = -1
    if pred <= 0.6:
        interval = 0
    elif pred > 0.6 and pred <= 0.7:
        interval = 1
    elif pred > 0.7 and pred <= 0.8:
        interval = 2
    elif pred > 0.8 and pred <= 0.9:
        interval = 3
    elif pred > 0.9 and pred <= 1.0:
        interval = 4
    return interval


def main():

    args = config.config_train()
    # mp.spawn(dist_singlegen,  nprocs=args.gpus, args=(args,))
    mp.spawn(dist_EL,  nprocs=args.gpus, args=(args,))
    mp.spawn(dist_train,  nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()
