# -*- coding: utf-8 -*-
import cv2,os,pdb
import numpy as np
# from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage import exposure

'''
# 去噪
ori_muti_path = r'./MUTI_NOISE/train/img/'
oroot = r'./MUTI_DENOISE_MEAN/train/img/'
os.makedirs(oroot,exist_ok=True)
img_list = os.listdir(ori_muti_path)
for name in tqdm(img_list):
    ipath = os.path.join(ori_muti_path,name)
    opath = os.path.join(oroot,name)
    img = cv2.imread(ipath)
    img_mean = cv2.blur(img, (5,5))
    cv2.imwrite(opath,img_mean)

ori_muti_path = r'./MUTI_NOISE/train/img/'
oroot = r'./MUTI_DENOISE_GAUSSIAN/train/img/'
os.makedirs(oroot,exist_ok=True)
img_list = os.listdir(ori_muti_path)
for name in tqdm(img_list):
    ipath = os.path.join(ori_muti_path,name)
    opath = os.path.join(oroot,name)
    img = cv2.imread(ipath)
    img_Guassian=cv2.GaussianBlur(img,(3,3),0)
    cv2.imwrite(opath,img_Guassian)

ori_muti_path = r'./MUTI_NOISE/train/img/'
oroot = r'./MUTI_DENOISE_MEDIA/train/img/'
os.makedirs(oroot,exist_ok=True)
img_list = os.listdir(ori_muti_path)
for name in tqdm(img_list):
    ipath = os.path.join(ori_muti_path,name)
    opath = os.path.join(oroot,name)
    img = cv2.imread(ipath)
    img_median = cv2.medianBlur(img, 5)
    cv2.imwrite(opath,img_median)

ori_muti_path = r'./MUTI_NOISE/train/img/'
oroot = r'./MUTI_DENOISE_BILATER/train/img/'
os.makedirs(oroot,exist_ok=True)
img_list = os.listdir(ori_muti_path)
for name in tqdm(img_list):
    ipath = os.path.join(ori_muti_path,name)
    opath = os.path.join(oroot,name)
    img = cv2.imread(ipath)
    img_bilater = cv2.bilateralFilter(img,9,75,75)
    cv2.imwrite(opath,img_bilater)
'''

# 修复光照 增强
# ori_muti_path = r'./MUTI_NOISE/train/img/'
# oroot = r'./MUTI_DELIGHT_CLAHE/train/img/'
# os.makedirs(oroot,exist_ok=True)
# img_list = os.listdir(ori_muti_path)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# for name in tqdm(img_list):
#     ipath = os.path.join(ori_muti_path,name)
#     opath = os.path.join(oroot,name)
#     img = cv2.imread(ipath)
#     dst = np.array([clahe.apply(img[:,:,0]),clahe.apply(img[:,:,1]),clahe.apply(img[:,:,2])])
#     dst = np.transpose(dst,(1,2,0))
#     cv2.imwrite(opath,dst)

# ori_muti_path = r'./MUTI_NOISE/train/img/'
# oroot = r'./MUTI_DELIGHT_EQUALHIST/train/img/'
# os.makedirs(oroot,exist_ok=True)
# img_list = os.listdir(ori_muti_path)
# for name in tqdm(img_list):
#     ipath = os.path.join(ori_muti_path,name)
#     opath = os.path.join(oroot,name)
#     img = cv2.imread(ipath)
#     equa = np.array([cv2.equalizeHist(img[:,:,0]),cv2.equalizeHist(img[:,:,1]),cv2.equalizeHist(img[:,:,2])])
#     equa = np.transpose(equa,(1,2,0))
#     cv2.imwrite(opath,equa)

ori_muti_path = r'./MUTI_NOISE/train/img/'
oroot = r'./MUTI_DELIGHT_GAMA5/train/img/'
os.makedirs(oroot,exist_ok=True)
img_list = os.listdir(ori_muti_path)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
for name in tqdm(img_list):
    ipath = os.path.join(ori_muti_path,name)
    opath = os.path.join(oroot,name)
    img = cv2.imread(ipath)
    out = exposure.adjust_gamma(img, 0.5)
    cv2.imwrite(opath,out)
