from easydict import EasyDict as edict
import os


# init
__C_AC = edict()

cfg_data = __C_AC

DATA_PATH = '/mnt/scratch/qingzhong/dataset/counting/audioCountingData/stage1'

__C_AC.STD_SIZE = (768, 1024)
__C_AC.TRAIN_SIZE = (576, 768)  # 2D tuple or 1D scalar
__C_AC.IMAGE_PATH = os.path.join(DATA_PATH, 'imgs')
__C_AC.DENSITY_PATH = os.path.join(DATA_PATH, 'density')
__C_AC.AUDIO_PATH = os.path.join(DATA_PATH, 'auds')

__C_AC.IS_CROSS_SCENE = False
__C_AC.IS_NOISE = False
__C_AC.IS_DENOISE = False
__C_AC.BRIGHTNESS = 1.0  # if is_noise, this param works
__C_AC.NOISE_SIGMA = 0  # if is_noise, this param works
__C_AC.LONGEST_SIDE = 512
__C_AC.BLACK_AREA_RATIO = 0
__C_AC.IS_RANDOM = False  # if is_noise, this param works

__C_AC.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])

__C_AC.LABEL_FACTOR = 1  # must be 1
__C_AC.LOG_PARA = 100.

__C_AC.RESUME_MODEL = ''  # model path
__C_AC.TRAIN_BATCH_SIZE = 48  # imgs

__C_AC.VAL_BATCH_SIZE = 1  # must be 1


