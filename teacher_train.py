

import os

from datetime import datetime
import argparse
from pathlib import Path

from cv2 import stereoCalibrate
from torchsummary.torchsummary import summary


from toolkits.data import FrameFaceDataset
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist


from toolkits import utils, split
from tqdm import tqdm
import shutil
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

from config import config
import time
from model import xception

###################
# 分布式训练
###################
# ! process marker
# * explaination marker
# ? testing code marker


def dist_train(gpu, args):
    # !step 1
    rank = gpu  # 当前进程号
    print('Rank id: ', rank)

    # !step 2
    # *将args获取的参数转化为变量

    train_datasets = args.traindb
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

    # ?max_train_samples = args.trainsamples
    log_interval = args.logint
    num_workers = args.workers
    seed = args.seed
    debug = args.debug

    # ?enable_attention = args.attention
    weights_folder = args.models_dir
    logs_folder = args.log_dir
    world_size = args.world_size
    backend = args.backend
    init_method = args.init_method
    epoch_run = args.epochs
    model_period = args.modelperiod
    tagnote = args.tagnote

    initial_model = args.index
    # suffix = args.suffix

    # initiate process group, decide the wat of process comunication
    dist.init_process_group(
        backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    torch.manual_seed(0)

    # gain model class form variables
    model_t_class = getattr(xception, args.net_t)

    model_t = model_t_class()
    transformer = utils.get_transformer(face_policy=face_policy, patch_size=face_size,
                                        net_normalizer=model_t.get_normalizer(), train=True)

    # generate tag
    tag = utils.make_train_tag(net_class=model_t_class,
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
        model_t.get_trainable_parameters(), lr=initial_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=patience,
        cooldown=2 * patience,
        min_lr=initial_lr*1e-7,
    )

    # !step4
    # *模型超参配置
    val_loss = min_val_loss = 10
    epoch = iteration = 0
    model_state = None
    opt_state = None

    # loading model

    # epoch, iteration = load_model(model_t, optimizer, path_list, mode,
    #                               initial_model)

    print(epoch)

    model_t = model_t.cuda(gpu)

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
        model_s = nn.SyncBatchNorm.convert_sync_batchnorm(model_t)
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

    if len(val_dataset_1) == 0:
        print('No validation samples. Halt.')
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
    val_loader = torch.utils.data.DataLoader(
        val_dataset_1, num_workers=num_workers, batch_size=batch_size, pin_memory=True,
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

            model_t.train()
            (batch_data, batch_labels), batch_paths = train_batch
            train_batch_num = len(batch_labels)

            train_num += train_batch_num

            start = time.time()
            # print(start-current)

            train_batch_loss, train_batch_pred = batch_forward(
                model_t, batch_data, batch_labels, batch_paths)
            # print(time.time()-start)

            _, train_pred_index = torch.max(train_batch_pred, 1)
            train_labels_list.append(batch_labels.numpy().flatten())
            train_pred_list.append(train_pred_index.cpu().numpy())

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

                train_acc = accuracy_score(train_labels, train_pred)

                tb.add_scalar('train/loss', train_loss, iteration)
                tb.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
                tb.add_scalar('epoch', epoch, iteration)
                tb.add_scalar('train/acc', train_acc, iteration)

                tb.flush()

                # save model every model_period
                if (iteration % model_period == 0):

                    # save_model_v2(model_t, optimizer, train_loss, val_loss,
                    #               iteration, batch_size, epoch, periodic_path.format(iteration))
                    save_model_v2(model_t, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, last_path)

                train_loss = train_num = 0
                train_labels_list = []
                train_pred_list = []

            # validate model every val_interval
            if iteration > 0 and (iteration % validation_interval == 0):

                # *Validation

                val_loss = validation_routine(
                    model_s, val_loader, 'val', iteration, tb)
                tb.flush()

                # adjust learning rate according to val_loss
                lr_scheduler.step(val_loss)

                # Model checkpoint
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_model_v2(model_s, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, bestval_path)
            # *每迭代一个batch +1
            iteration = iteration + 1
            # current = time.time()
        epoch = epoch + 1


'''
method definition:
------------------
:param model        training model
:param data         training data (Batch_size,3,299,299)
:param labels       labels corresponding to data
-----------------
:return loss[float]          training loss   
:return pred[ndarray]        training prediction
'''


def batch_forward(model: nn.Module, data: torch.Tensor, labels: torch.Tensor, paths: list[Path]):

    # correct,total = 0,0
    data = data.float().cuda(non_blocking=True)

    labels = labels.cuda(non_blocking=True)

    pred = model(data)
    pred = nn.Sigmoid()(pred)
    # _, predicted = torch.max(pred, 1)
    # correct += (predicted == labels).sum().item()
    # total = len(labels)
    # print(correct/total)
    # labels = labels.long().squeeze()
    # loss = nn.CrossEntropyLoss()(pred, labels)

    # pred = torch.sigmoid(pred)

    # t_out = torch.sigmoid(t_out)
    # # # 将网络的输出转化为[0,1]，同时转为nadarray
    # pred = pred.detach().cpu().numpy()
    # t_out = t_out.detach().cpu().numpy()
    # # 计算Loss
    # pred = pred.detach().cpu().numpy()
    # train_predL = [int(item > 0.2) for item in pred]
    # for i in range(len(train_predL)):
    #     if(labels[i] != train_predL[i] ):
    #         # print(paths[i])

    loss = nn.CrossEntropyLoss()(pred, labels.long().flatten())
    # pred = nn.Sigmoid()(pred)

    # pred = pred.detach().cpu().numpy()

    # loss = Loss_cal(pred, t_out, w, labels, iteration, tb, flag)

    return loss, pred


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


def validation_routine(net, val_loader, tag: str, iteration, tb, loader_len_norm: int = None):
    # switch to eval mode
    net.eval()

    loader_len_norm = loader_len_norm if loader_len_norm is not None else val_loader.batch_size
    val_num = 0
    val_loss = 0.
    val_labels_list = []
    val_pred_list = []
    for val_data in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader)):

        (batch_data, batch_labels), batch_paths = val_data
        # 给定batch大小
        val_batch_num = len(batch_labels)

        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(net, batch_data,
                                                           batch_labels, batch_paths)
        _, val_pred_index = torch.max(val_batch_pred, 1)

        val_labels_list.append(batch_labels.numpy().flatten())
        val_pred_list.append(val_pred_index.cpu().numpy())

        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss /= val_num
    tb.add_scalar('{}/loss'.format(tag), val_loss, iteration)

    val_labels = np.concatenate(val_labels_list)
    val_pred = np.concatenate(val_pred_list)

    acc = accuracy_score(val_labels, val_pred)
    tb.add_scalar('{}/acc'.format(tag), acc, iteration)

    return val_loss


def main():

    args = config.config_teacher()
    mp.spawn(dist_train,  nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()
