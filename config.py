from easydict import EasyDict
import time

# init
datasets = {'ShanghaiTech Part_A':  'SHHA',
            'ShanghaiTech Part_B':  'SHHB',
            'UCF_CC_50':            'UCF50',
            'UCF-QNRF':             'QNRF'}
methods = {'CSRNet': 'CSRNet',
           'NewNet': 'NewNet',
           'PSNet':  'PSNet',
           'TransNet': 'TransNet'}
__C = EasyDict()
cfg = __C

# ------------------------------TRAIN------------------------
__C.SEED = 3035  # random seed,  for reproduction
__C.DATASET = datasets['ShanghaiTech Part_A']  # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD
__C.NET = methods['TransNet']  # net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet
__C.GPU_ID = [1]  # single gpu: [0], [1], ...; multi gpus: [0, 1, ...]


# learning rate settings
__C.LR = 1e-5  # learning rate
__C.WEIGHT_DECAY = 5e-4  # weight decay
__C.LR_DECAY = 0.995  # decay rate
__C.LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 1  # decay frequency
__C.MAX_EPOCH = 400

__C.RESUME = False  # continue training
if __C.RESUME:
    __C.RESUME_PATH = './exp/04-25_09-19_SHHB_VGG_1e-05/latest_state.pth'  #

# The proposed method
if __C.NET == 'NewNet':
    __C.NET_MULTI_FUSE = False
    __C.NET_IS_VIT = True
    __C.PATCH_SIZE = 16

if __C.NET == 'TransNet':
    __C.GRID_SIZE = 4
    __C.SPAN_RANGE = 0.9

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on
if __C.NET == 'SANet' or __C.NET == 'CMTL':
    __C.LAMBDA_1 = 1e-4  # SANet:0.001 CMTL 0.0001

# print
__C.PRINT_FREQ = 5

now = time.strftime("%Y%m%d-%H%M%S", time.localtime())

__C.EXP_NAME = __C.NET \
               + '-' + __C.DATASET \
               + '-' + now \
               # + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':  # only for UCF50
    from datasets.UCF50.setting import cfg_data
    __C.VAL_INDEX = cfg_data.VAL_INDEX
    __C.EXP_NAME += '_' + str(__C.VAL_INDEX)

if __C.DATASET == 'GCC':  # only for GCC
    from datasets.GCC.setting import cfg_data
    __C.VAL_MODE = cfg_data.VAL_MODE
    __C.EXP_NAME += '_' + __C.VAL_MODE

# __C.PRE_GCC = False  # use the pretrained model on GCC dataset
# if __C.PRE_GCC:
#     __C.PRE_GCC_MODEL = 'path to model'  # path to model

__C.EXP_PATH = './exp'  # the path of logs, checkpoints, and current codes
__C.EXCEPTION = ['exp', 'logs', 'test_results', 'results_reports', '__pycache__']  # dir or file not to backup

# ------------------------------VAL------------------------
__C.VAL_FREQ = 10  # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ
__C.VAL_DENSE_START = 50

# ------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1  # must be 1 for training images with the different sizes

# ------------------------------PATCH------------------------
if __C.NET == 'AMRNet':
    __C.PATCHMAX = {
        'SHHA': 100,
        'SHHB': 30,
        'QNRF': 100,
        'UCF50': 100,
        'UCSD': 30,
        'Mall': 30,
        'FDST': 30
    }

# ================================================================================
# ================================================================================
# ================================================================================
