import cv2
import argparse
import os


def change_brightness(img, value):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2]*value
    
    hsv[hsv > 255] = 255
    hsv[hsv < 0] = 0
    
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return img


if __name__ == '__main__':
    # img = cv2.imread('061.jpg')
    # img = change_brightness(img, 0.5) # lower is darker
    # cv2.imwrite("image_processed.jpg", img)

    args = argparse.ArgumentParser()
    args.add_argument('--image_dir', type=str, default='')
    args.add_argument('--save_dir', type=str, default='')
    args.add_argument('--value', type=float, default=0.5)
    opt = args.parse_args()
    file_list = os.listdir(opt.image_dir)
    for f in file_list:

        try:
            print('OK ' + f)
            img = cv2.imread(os.path.join(opt.image_dir, f))
            img = change_brightness(img, opt.value)
            cv2.imwrite(os.path.join(opt.save_dir, f), img)
        except:
            print('Fail ' + f)
            pass
