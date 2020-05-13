#coding:utf-8
import scipy.io as scio
import numpy as np
from PIL import Image
from scipy import misc as smisc
import argparse
import os
import imageio


def gaussian_filter(shape=[3, 3], sigma=1.0):
    """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return h


def density_map(k_size, sigma, im_file, points, imgid):
    """
    :param k_size: Gaussian kernel size
    :param sigma:
    :param im_file: image file
    :param points: array, N x 2
    :return:
    """
    image = imageio.imread(im_file)
    im_sz = np.shape(image)

    # assert im_sz == 3  # RGB
    h = im_sz[0]
    w = im_sz[1]

    im_density = np.zeros(shape=[h, w])
    H = gaussian_filter(shape=[k_size, k_size], sigma=sigma)
    hk_size = int(k_size / 2)
    for j in range(points.shape[0]):
        x, y = map(int, points[j])
        if x == w: x -= 1
        if y == h: y -= 1
        if x < 0 or y < 0 or x > w or y > h:
            continue
        min_img_x = max(0, x - hk_size)
        min_img_y = max(0, y - hk_size)
        max_img_x = min(x + hk_size + 1, w - 1)
        max_img_y = min(y + hk_size + 1, h - 1)

        kernel_x_min = (hk_size - x if x <= hk_size else 0)
        kernel_y_min = (hk_size - y if y <= hk_size else 0)
        kernel_x_max = kernel_x_min + max_img_x - min_img_x
        kernel_y_max = kernel_y_min + max_img_y - min_img_y

        im_density[min_img_y:max_img_y, min_img_x:max_img_x] += H[kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max]
    return im_density


def show_example(args, image_id, density):
    imageio.imsave(os.path.join(args.example_dir, str(image_id)+'.jpg'), density)


def main(args):
    label_dir = args.label_dir
    image_dir = args.image_dir
    save_dir = args.save_dir
    label_files = os.listdir(label_dir)
    print('Number of labeled images: %d' % len(label_files))
    for label_file in label_files:
        print(label_file)
        if '._' in label_file:
            continue
        image_id = label_file.split('.')[0].split('_')[1]
        image_file = os.path.join(image_dir, image_id + '.jpg')
        with open(os.path.join(label_dir, label_file), 'r') as f:
            raw_labels = f.read()
        raw_labels = raw_labels.strip().split('\n')
        labels = np.zeros([len(raw_labels), 2])
        for i in range(len(raw_labels)):
            xy = raw_labels[i]
            # print(xy)
            try:
                x, y = xy.strip().split(' ')
            except:
                continue
            labels[i, 0] = int(x)
            labels[i, 1] = int(y)
        im_density = density_map(k_size=args.k_size, sigma=args.sigma, im_file=image_file, points=labels, imgid=image_id)
        prob = np.random.uniform()
        if prob < args.prob_train:
            scio.savemat(os.path.join(save_dir, 'train', image_id + '.mat'), {'map': im_density})
        else:
            scio.savemat(os.path.join(save_dir, 'test', image_id + '.mat'), {'map': im_density})
        # assert abs(im_density.sum() - len(raw_labels)) < 1
        show_example(args, image_id, 255 / (im_density.max() - im_density.min()) * (im_density - im_density.min()))
        print('%s Ok, NO. of people: %d, density map: %.5f' % (image_id, len(raw_labels), im_density.sum()))
        

if __name__ == '__main__':
    # im_file = 'imgs_0025.jpg'
    # with open(os.path.join('160labels', 'new_0025.txt'), 'r') as f:
    #     raw_labels = f.read()
    # raw_labels = raw_labels.strip().split('\n')
    # labels = np.zeros([len(raw_labels), 2])
    # for i in range(len(raw_labels)):
    #     xy = raw_labels[i]
    #     x, y = xy.strip().split(' ')
    #     labels[i, 0] = int(x)
    #     labels[i, 1] = int(y)
    # den = density_map(3.0, 1.0, im_file, labels)
    # import matplotlib.pyplot as plt
    # plt.imsave('show.png', den)
    # plt.show()
    args = argparse.ArgumentParser()
    args.add_argument('--label_dir', type=str, default=r'D:\daily-info\temp\baidu\160labels', help='dir of label files')
    args.add_argument('--image_dir', type=str, default=r'\\169.255.73.1\gjy\人群计数原版视频数据\最终版整理数据集\processedDataStage1\imgs', help='dir of images')
    args.add_argument('--save_dir', type=str, default='.\save', help='dir to save mat files')
    args.add_argument('--example_dir', type=str, default='audioCountingData/stage1/example')
    args.add_argument('--prob_train', type=float, default=0.8)
    args.add_argument('--sigma', type=float, default=4.0)
    args.add_argument('--k_size', type=float, default=15.0)
    opt = args.parse_args()
    main(opt)

