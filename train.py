import os
import numpy as np
import torch
import argparse
import time

from config import cfg

# args = argparse.ArgumentParser()
# args.add_argument('--net_name', type=str, default='', help='name of net')
# args.add_argument('--resume', type=int, default=0, help='whether to resume model')
# args.add_argument('--resume_path', type=str, default='')
# args.add_argument('--settings', type=str, default='')
#
# args.add_argument('--is_noise', type=int, default=0)
# args.add_argument('--brightness', type=float, default=1.0)
# args.add_argument('--noise_sigma', type=float, default=25)
# args.add_argument('--longest_side', type=int, default=1024)
# args.add_argument('--black_area_ratio', type=float, default=0)
# args.add_argument('--is_random', type=int, default=0)
#
# opt = args.parse_args()
#
# cfg.NET = opt.net_name
# cfg.RESUME = (opt.resume == 1)
# cfg.RESUME_PATH = os.path.join('../trained_models/exp', opt.resume_path + '/latest_state.pth')
# cfg.SETTINGS = opt.settings
#
# now = time.strftime("%m-%d_%H-%M", time.localtime())
# cfg.EXP_NAME = cfg.SETTINGS + '_' + cfg.DATASET + '_' + cfg.NET + '_' + str(cfg.LR)


#------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True


#------------prepare data loader------------
data_mode = cfg.DATASET
if data_mode is 'SHHA':
    from datasets.SHHA.loading_data import loading_data 
    from datasets.SHHA.setting import cfg_data 
elif data_mode is 'SHHB':
    from datasets.SHHB.loading_data import loading_data 
    from datasets.SHHB.setting import cfg_data 
elif data_mode is 'QNRF':
    from datasets.QNRF.loading_data import loading_data 
    from datasets.QNRF.setting import cfg_data 
elif data_mode is 'UCF50':
    from datasets.UCF50.loading_data import loading_data 
    from datasets.UCF50.setting import cfg_data 
elif data_mode is 'WE':
    from datasets.WE.loading_data import loading_data 
    from datasets.WE.setting import cfg_data 
elif data_mode is 'GCC':
    from datasets.GCC.loading_data import loading_data
    from datasets.GCC.setting import cfg_data
elif data_mode is 'Mall':
    from datasets.Mall.loading_data import loading_data
    from datasets.Mall.setting import cfg_data
elif data_mode is 'UCSD':
    from datasets.UCSD.loading_data import loading_data
    from datasets.UCSD.setting import cfg_data
elif data_mode is 'AC':  # Qingzhong
    from datasets.AC.loading_data import loading_data
    from datasets.AC.setting import cfg_data

    # cfg_data.IS_NOISE = (opt.is_noise == 1)
    # cfg_data.BRIGHTNESS = opt.brightness
    # cfg_data.NOISE_SIGMA = opt.noise_sigma
    # cfg_data.LONGEST_SIDE = opt.longest_side
    # cfg_data.BLACK_AREA_RATIO = opt.black_area_ratio
    # cfg_data.IS_RANDOM = (opt.is_random == 1)

print(cfg, cfg_data)


#------------Prepare Trainer------------
net = cfg.NET
if net in ['MCNN', 'AlexNet', 'VGG', 'VGG_DECODER', 'Res50', 'Res101', 'CSRNet','Res101_SFCN',
           'CSRNet_IN', 'CSRNet_Audio', 'CANNet', 'CANNet_Audio', 'CSRNet_Audio_Concat', 'CANNet_Audio_Concat',
           'CSRNet_Audio_Guided', 'CANNet_Audio_Guided'
           ]:
    from trainer import Trainer
elif net in ['SANet', 'SANet_Audio']:
    from trainer_for_M2TCC import Trainer # double losses but signle output
elif net in ['CMTL']: 
    from trainer_for_CMTL import Trainer # double losses and double outputs
elif net in ['PCCNet']:
    from trainer_for_M3T3OCC import Trainer

#------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(loading_data, cfg_data, pwd)
cc_trainer.forward()
