import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps, ImageEnhance
import librosa
from .audioset import vggish_input
import cv2


class AC(data.Dataset):
    def __init__(self, img_path, den_path, aud_path, mode, main_transform=None, img_transform=None, gt_transform=None,
                 is_noise=False, brightness_decay=0.3, noise_sigma=25, longest_side=1024, black_area_ratio=0,
                 is_random=True, is_denoise=False):
        self.img_path = img_path
        self.gt_path = den_path
        self.aud_path = aud_path
        # self.data_files = [filename for filename in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, filename))]
        self.image_ids = [filename.split('.')[0] for filename in os.listdir(self.gt_path)]
        self.num_samples = len(self.image_ids)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        self.is_noise = is_noise
        self.brightness_decay = brightness_decay
        self.noise_sigma = noise_sigma
        self.longest_side = longest_side
        self.black_area_ratio = black_area_ratio
        self.is_random = is_random
        self.is_denoise = is_denoise

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img, den, aud = self.read_image_and_gt(image_id)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den, torch.FloatTensor(aud)

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, image_id):
        longest_size = self.longest_side
        img = Image.open(os.path.join(self.img_path, image_id+'.jpg'))
        if img.mode == 'L':
            img = img.convert('RGB')
        if self.is_noise:
            if self.is_random:
                img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0, self.brightness_decay))
                img = self.add_gaussian_noise(img, mean=0, max_sigma=self.noise_sigma)
            else:
                img = ImageEnhance.Brightness(img).enhance(self.brightness_decay)
                img = self.add_gaussian_noise(img, mean=0, max_sigma=self.noise_sigma)
        if self.is_denoise:
            img = self.gaussian_blur_denoise(img)
        img = self.random_black(img, self.black_area_ratio)
        w, h = img.size
        if w > h:
            factor = w / longest_size
            img = img.resize((longest_size, int(h / factor)), Image.BICUBIC)
        else:
            factor = h / longest_size
            img = img.resize((int(w / factor), longest_size), Image.BICUBIC)

        den = sio.loadmat(os.path.join(self.gt_path, image_id + '.mat'))
        den = den['map']
        # den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values
        num_people = den.sum()
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        if w > h:
            den = np.array(den.resize((longest_size, int(h / factor)), Image.BICUBIC)) * factor * factor
        else:
            den = np.array(den.resize((int(w / factor), longest_size), Image.BICUBIC)) * factor * factor
        den = Image.fromarray(den)
        # print(num_people, np.array(den).sum())
        # assert abs(np.array(den).sum()-num_people) < 1

        aud, sr = librosa.load(os.path.join(self.aud_path, image_id + '.wav'), sr=None, mono=True)
        log_mel_map = vggish_input.waveform_to_examples(aud, sample_rate=sr)
        return img, den, log_mel_map

    def get_num_samples(self):
        return self.num_samples

    def add_gaussian_noise(self, image, mean, max_sigma):
        image = np.array(image).astype(float)
        row, col, channel = image.shape
        mean = 0
        if self.is_random:
            sigma = np.sqrt(np.random.uniform() * np.square(max_sigma / 255.))
        else:
            sigma = np.sqrt(np.square(max_sigma / 255.))
        gauss = np.random.normal(mean, sigma, (row, col, channel))
        noise = np.round(gauss * 255)
        image += noise
        image[image > 255] = 255
        image[image < 0] = 0
        return Image.fromarray(np.uint8(image))

    def random_black(self, image, ratio):
        if ratio < 0:
            ratio = 0
        if ratio > 1:
            ratio = 1
        if ratio == 0:
            return image
        image = np.array(image).astype(float)
        row, col, channel = image.shape
        if ratio == 1:
            return Image.fromarray(np.uint8(np.zeros([row, col, channel])))
        r = np.sqrt(ratio)
        black_area_row = int(row * r)
        black_area_col = int(col * r)
        remain_row = row - black_area_row
        remain_col = col - black_area_col
        x = np.random.randint(low=0, high=remain_row)
        y = np.random.randint(low=0, high=remain_col)
        image[x:(x + black_area_row), y:(y + black_area_col), :] = np.zeros([black_area_row, black_area_col, channel])
        return Image.fromarray(np.uint8(image))

    def gaussian_blur_denoise(self, image):
        image = np.array(image).astype(float)
        image = cv2.GaussianBlur(image, (11, 11), 0)
        w, h, c = image.shape
        assert c == 3
        for i in range(c):
            image[:, :, i] = cv2.equalizeHist(image[:, :, i].astype(np.uint8))

        return Image.fromarray(np.uint8(image))



