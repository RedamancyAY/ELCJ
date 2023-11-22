from distutils.command.config import config
import teacher_train
import CJEL_train
import CJEL_gen
import model_gen
from config import config

import torch.multiprocessing as mp


# *train a teacher at a time
def singleT_train():
    args = config.config_teacher()
    mp.spawn(teacher_train.dist_train,  nprocs=args.gpus, args=(args,))
    return 0

# *train four teachers in sequence


def multiT_train():
    tagnotes = ['DF_T1', 'F2F_T1', 'FS_T1', 'NT_T1', 'Celeb_T1']

    for i in range(2, 4):
        args = config.config_teacher()
        args.tagnote = tagnotes[i]
        args.trainIndex = i
        mp.spawn(teacher_train.dist_train,  nprocs=args.gpus, args=(args,))
    return 0


def multiCJELS_train():
    tagnotes = ['DF_EL2', 'F2F_EL2',
                'FS_EL2', 'NT_EL2']
    # tagnotes = ['DF_normalKD', 'F2F_normalKD', 'FS_normalKD']
    net_t_paths = ['/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-DF_T1/last.pth',
                   '/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-F2F_T1/last.pth',
                   '/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-FS_T1/last.pth',
                   '/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-NT_T1/last.pth'
                   ]
    for i in range(0, 3):
        args = config.config_train()
        args.tagnote = tagnotes[i]
        args.net_t_path = net_t_paths[i]
        args.trainIndex = i
        mp.spawn(CJEL_train.dist_train,  nprocs=args.gpus, args=(args,))

    return 0


def modelgen():
    tagnotes = ['FS-DFDCB_complete1_4']
    netpath = '/mnt/8T/hou/multicard_CSFEL/weights/binclass/net-WholeNet_traindb-ff-c23-720-140-140_face-scale_size-299_seed-43_note-FS_complete1/last.pth'
    val_index1 = [2]
    val_index2 = [3]
    trainindexs = [3]
    for i in range(1):
        args = config.config_gen()
        args.tagnote = tagnotes[i]
        args.net_s_path = netpath
        args.trainIndex = trainindexs[i]
        args.valIndex_1 = val_index1[i]
        args.valIndex_2 = val_index2[i]
        mp.spawn(CJEL_gen.dist_gen,  nprocs=args.gpus, args=(args,))


def othermodel_gen():
    tagnotes = ['FS-DFDCB_']
    netpath = '/mnt/8T/hou/multicard_v1.3/weights/binclass/net-MesoInception4_traindb-ff-c23-720-140-140_face-scale_size-299_seed-1_note-MesoInception4_FS_2/bestval.pth'
    val_index1 = [2]
    val_index2 = [3]
    trainindexs = [3]
    for i in range(1):
        args = config.config_gen()
        args.tagnote = tagnotes[i]
        args.net_s_path = netpath
        args.trainIndex = trainindexs[i]
        args.valIndex_1 = val_index1[i]
        args.valIndex_2 = val_index2[i]
        mp.spawn(model_gen.dist_gen,  nprocs=args.gpus, args=(args,))


if __name__ == '__main__':

    # pick your train preference
    # singleT_train()
    # multiT_train()
    # multiCJELS_train()
    # modelgen()
    # othermodel_gen()
