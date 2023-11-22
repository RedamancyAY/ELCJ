import ml_collections


def config_train():
    config = ml_collections.ConfigDict()
    # !random parameter setting
    config.seed = 43

    # !distributed training setting
    config.gpus = 2
    config.world_size = 2
    config.backend = 'nccl'
    config.init_method = 'tcp://10.249.178.201:12345'
    config.syncbn = True

    # !transform setting
    config.face = 'scale'
    config.size = 299

    # !training parameter setting
    config.debug = False
    config.logint = 100
    config.modelperiod = 500
    config.valint = 500

    config.batch = 16
    config.batch_train = 16
    config.epochs = 20
    config.net_s = 'WholeNet'
    config.net_t = 'xception'
    config.net_t_path = "/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-test/bestval.pth"
    config.traindb = ["ff-c23-720-140-140"]
    config.trainIndex = 0
    config.tagnote = 'DF_v3'

    # !dataset setting
    config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/df/output/FFPP_df.pkl'
    config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output'
    config.workers = 4

    # !optimizer setting
    config.lr = 1e-4
    config.patience = 10

    # !model loading setting
    config.models_dir = '/mnt/8T/hou/multicard_CSFEL/weights/binclass/'
    config.mode = 1
    config.index = 0

    # !log setting
    config.log_dir = '/mnt/8T/hou/multicard_CSFEL/runs/binclass/'

    return config


def config_teacher():
    config = ml_collections.ConfigDict()
    # !random parameter setting
    config.seed = 22

    # !distributed training setting
    config.gpus = 2
    config.world_size = 2
    config.backend = 'nccl'
    config.init_method = 'tcp://10.249.178.201:1234'
    config.syncbn = True

    # !transform setting
    config.face = 'scale'
    config.size = 299

    # !training parameter setting
    config.debug = False
    config.logint = 100
    config.modelperiod = 500
    config.valint = 100

    config.batch = 64
    config.epochs = 15
    config.net_s = 'WholeNet'
    config.net_t = 'xception'
    config.net_t_path = "/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-test/bestval.pth"
    config.traindb = ["ff-c23-720-140-140"]
    config.trainIndex = 0
    config.tagnote = 'F2F_T4'

    # !dataset setting
    config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/df/output/FFPP_df.pkl'
    config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output'
    config.workers = 4

    # !optimizer setting
    config.lr = 1e-4
    config.patience = 10

    # !model loading setting
    config.models_dir = '/mnt/8T/hou/multicard_teacher/weights/binclass/'
    config.mode = 1
    config.index = 0

    # !log setting
    config.log_dir = '/mnt/8T/hou/multicard_teacher/runs/binclass/'

    return config


def config_gen():
    config = ml_collections.ConfigDict()
    # !random parameter setting
    config.seed = 22

    # !distributed training setting
    config.gpus = 2
    config.world_size = 2
    config.backend = 'nccl'
    config.init_method = 'tcp://10.249.178.201:1234'
    config.syncbn = True

    # !transform setting
    config.face = 'scale'
    config.size = 299

    # !training parameter setting
    config.debug = False
    config.logint = 40
    config.modelperiod = 100
    config.valint = 100

    config.batch = 32
    config.epochs = 10
    # config.net_s = 'WholeNet'
    config.net_s = 'WholeNet'
    config.net_t = 'WholeNet'
    config.net_t_path = "/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-test/bestval.pth"
    config.net_s_path = "/mnt/8T/hou/multicard_gen/weights/binclass/net-WholeNet_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-Gtest2_04_2/it002000.pth"
    # config.net_s_path = "/mnt/8T/hou/multicard_gen/weights/binclass/net-WholeNet_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-Gtest2_01_9/it000000.pth"

    # config.net_s_path = "/mnt/8T/hou/multicard_gen/weights/binclass/net-WholeNet_traindb-ff-c23-500-140-140_face-scale_size-299_seed-22_note-Gtest2_01_6/it003000.pth"
    config.traindb1 = ["ff-c23-500-140-140"]
    config.traindb2 = ["celebdf-500-50-40"]
    config.trainIndex = 2
    config.valIndex_1 = 0
    config.valIndex_2 = 1
    config.valIndex_3 = 2
    config.tagnote = 'Gtest2_0C_2'

    # !dataset setting
    config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/df/output/FFPP_df.pkl'
    config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output'
    config.celebdf_faces_df_path = '/mnt/8T/hou/celeb-df/faces/celeb_df.pkl'
    config.celebdf_faces_dir = '/mnt/8T/hou/celeb-df/faces'
    config.workers = 4

    # !optimizer setting
    config.lr = 1e-4
    config.patience = 10

    # !model loading setting
    config.models_dir = '/mnt/8T/hou/multicard_gen/weights/binclass/'
    config.mode = 1
    config.index = 0

    # !log setting
    config.log_dir = '/mnt/8T/hou/multicard_gen/runs/binclass/'

    return config


def config_FReTAL():
    config = ml_collections.ConfigDict()
    # !随机参数配置
    config.seed = 1

    # !硬件配置
    # 使用的GPU数目
    config.gpus = 2
    config.world_size = 2
    # 使用的进程平台
    config.backend = 'nccl'
    #
    config.init_method = 'tcp://10.249.178.201:34567'
    # !transform配置
    config.face = 'scale'
    config.size = 299
    # !训练参数配置
    config.batch = 384
    config.batch_val = 384
    config.epochs = 30
    config.syncbn = True
    # 空域分支选用
    config.net_s = 'xception'
    config.net_t = 'ShallowNet_ensemble'
    # 训练集与验证集切分标准
    config.traindb = ["ff-c23-20-140"]
    config.valdb = ["ff-c23-720-140-140"]
    # 训练过程中采取的学习率震荡参数
    config.vibperiod = 5000
    config.vibfactor = 4
    # !数据集配置
    # 切割脸部照片的存放目录
    # config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/faces/df/output/c40.pkl'
    config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/df/output/FFPP_df.pkl'
    # 切割脸部的Dataframe存放地点
    # config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output/directory1'
    config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output'
    # 多久验证一次模型，单位（batch）
    config.valint = 100
    config.valsamples = 6000
    # 多久记录一次log，单位（batch）
    config.logint = 100
    # 多久保存一次模型
    config.modelperiod = 500
    # !优化器配置
    config.lr = 1e-4
    config.patience = 5
    # !模型加载配置
    config.scratch = False
    config.models_dir = '/mnt/8T/hou/multicard_v1.3/weights/binclass/'
    # 0会加载最优模型，1会加载最新的模型，2会加载制定的模型
    config.mode = 1
    config.index = 0
    config.workers = 4

    # !logpath
    config.log_dir = '/mnt/8T/hou/multicard_v1.3/runs/binclass/'

    # !暂时无用配置
    config.debug = False
    config.dfdc_faces_df_path = ''
    config.dfdc_faces_dir = ''

    return config


def config_ShallowNet():
    config = ml_collections.ConfigDict()
    # !随机参数配置
    config.seed = 1

    # !硬件配置
    # 使用的GPU数目
    config.gpus = 2
    config.world_size = 2
    # 使用的进程平台
    config.backend = 'nccl'
    #
    config.init_method = 'tcp://10.249.178.201:34567'
    # !transform配置
    config.face = 'scale'
    config.size = 299
    # !训练参数配置
    config.batch = 128
    config.batch_val = 128
    config.epochs = 30
    config.syncbn = True
    # 空域分支选用
    config.net_s = 'ShallowNet_ensemble'
    config.net_t = 'ShallowNet_ensemble'
    # 训练集与验证集切分标准
    config.traindb = ["ff-c23-500-140"]
    config.valdb = ["ff-c23-720-140-140"]
    # 训练过程中采取的学习率震荡参数
    config.vibperiod = 5000
    config.vibfactor = 4
    # !数据集配置
    # 切割脸部照片的存放目录
    # config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/faces/df/output/c40.pkl'
    config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/df/output/FFPP_df.pkl'
    # 切割脸部的Dataframe存放地点
    # config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output/directory1'
    config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output'
    # 多久验证一次模型，单位（batch）
    config.valint = 250
    config.valsamples = 6000
    # 多久记录一次log，单位（batch）
    config.logint = 100
    # 多久保存一次模型，
    config.modelperiod = 500
    # !优化器配置
    config.lr = 1e-3
    config.patience = 5
    # !模型加载配置
    config.scratch = False
    config.models_dir = '/mnt/8T/hou/multicard_gen/weights/binclass/'
    # 0会加载最优模型，1会加载最新的模型，2会加载制定的模型
    config.mode = 1
    config.index = 0
    config.workers = 4

    # !logpath
    config.log_dir = '/mnt/8T/hou/multicard_gen/runs/binclass/'

    # !暂时无用配置
    config.debug = False
    config.dfdc_faces_df_path = ''
    config.dfdc_faces_dir = ''

    return config


def config_SingleGen():
    config = ml_collections.ConfigDict()
    # !随机参数配置
    config.seed = 1

    # !硬件配置
    # 使用的GPU数目
    config.gpus = 2
    config.world_size = 2
    # 使用的进程平台
    config.backend = 'nccl'
    #
    config.init_method = 'tcp://10.249.178.201:34567'
    # !transform配置
    config.face = 'scale'
    config.size = 299
    # !训练参数配置
    config.batch = 64
    config.batch_val = 64
    config.epochs = 20
    config.syncbn = True
    # 空域分支选用
    config.net_s = 'xception'
    config.net_t = 'ShallowNet_ensemble'
    # 训练集与验证集切分标准
    config.traindb = ["ff-c23-100-140"]
    config.valdb = ["ff-c23-720-140-140"]
    # 训练过程中采取的学习率震荡参数
    config.vibperiod = 5000
    config.vibfactor = 4
    # !数据集配置
    # 切割脸部照片的存放目录
    # config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/faces/df/output/c40.pkl'
    config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/df/output/FFPP_df.pkl'
    # 切割脸部的Dataframe存放地点
    # config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output/directory1'
    config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output'
    # 多久验证一次模型，单位（batch）
    config.valint = 267
    config.valsamples = 6000
    # 多久记录一次log，单位（batch）
    config.logint = 50
    # 多久保存一次模型，
    config.modelperiod = 500
    # !优化器配置
    config.lr = 1e-4
    config.patience = 5
    # !模型加载配置
    config.scratch = False
    config.models_dir = '/mnt/8T/hou/multicard_gen/weights/binclass/'
    # 0会加载最优模型，1会加载最新的模型，2会加载制定的模型
    config.mode = 1
    config.index = 0
    config.workers = 4

    # !logpath
    config.log_dir = '/mnt/8T/hou/multicard_gen/runs/binclass/'

    # !暂时无用配置
    config.debug = False
    config.dfdc_faces_df_path = ''
    config.dfdc_faces_dir = ''

    return config


def config_Mesotrain():
    config = ml_collections.ConfigDict()
    # !随机参数配置
    config.seed = 1

    # !硬件配置
    # 使用的GPU数目
    config.gpus = 2
    config.world_size = 2
    # 使用的进程平台
    config.backend = 'nccl'
    #
    config.init_method = 'tcp://10.249.178.201:34567'
    # !transform配置
    config.face = 'scale'
    config.size = 299
    # !训练参数配置
    config.batch_t = 128
    config.batch_v = 128
    config.epochs = 30
    config.syncbn = True
    # 空域分支选用
    config.net_s = 'MesoInception4'
    config.net_t = 'MesoInception4'
    # 训练集与验证集切分标准
    config.traindb = ["ff-c23-720-140-140"]
    config.valdb = ["ff-c23-720-140-140"]
    # 训练过程中采取的学习率震荡参数
    config.vibperiod = 5000
    config.vibfactor = 4
    # !数据集配置
    # 切割脸部照片的存放目录
    # config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/faces/df/output/c40.pkl'
    config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/df/output/FFPP_df.pkl'
    # 切割脸部的Dataframe存放地点
    # config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output/directory1'
    config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output'
    # 多久验证一次模型，单位（batch）
    config.valint = 360
    config.valsamples = 6000
    # 多久记录一次log，单位（batch）
    config.logint = 100
    # 多久保存一次模型，
    config.modelperiod = 360
    # !优化器配置
    config.lr = 1e-3
    config.patience = 5
    # !模型加载配置
    config.scratch = False
    config.models_dir = '/mnt/8T/hou/multicard_v1.3/weights/binclass/'
    # 0会加载最优模型，1会加载最新的模型，2会加载制定的模型
    config.mode = 1
    config.index = 0
    config.workers = 4

    # !logpath
    config.log_dir = '/mnt/8T/hou/multicard_v1.3/runs/binclass/'

    # !暂时无用配置
    config.debug = False
    config.dfdc_faces_df_path = ''
    config.dfdc_faces_dir = ''

    return config
