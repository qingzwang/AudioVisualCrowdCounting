import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import pdb
import matplotlib.pyplot as pyplot


class Tester():
    def __init__(self, dataloader, cfg_data, pwd):

        self.save_path = os.path.join('/mnt/home/dongsheng/hudi/counting/trained_models/img-den-mel-pred',
                                      str(cfg.NET) + '-' + 'noise-' + str(cfg_data.IS_NOISE) + '-' + str(
                                          cfg_data.BRIGHTNESS) +
                                      '-' + str(cfg_data.NOISE_SIGMA) + '-' + str(cfg_data.LONGEST_SIDE) + '-' + str(
                                          cfg_data.BLACK_AREA_RATIO) +
                                      '-' + str(cfg_data.IS_RANDOM) + '-' + 'denoise-' + str(cfg_data.IS_DENOISE))
        if not os.path.exists(self.save_path):
            os.system('mkdir ' + self.save_path)
        else:
            os.system('rm -rf ' + self.save_path)
            os.system('mkdir ' + self.save_path)

        self.cfg_data = cfg_data
        self.cfg = cfg

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        self.net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()
        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.CCN.parameters(), cfg.LR, momentum=0.9, weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)

        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.epoch = 0
        self.i_tb = 0

        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        self.train_loader, self.val_loader, self.test_loader, self.restore_transform = dataloader()

        if cfg.RESUME:
            # latest_state = torch.load(cfg.RESUME_PATH)
            # self.net.load_state_dict(latest_state['net'])
            # self.optimizer.load_state_dict(latest_state['optimizer'])
            # self.scheduler.load_state_dict(latest_state['scheduler'])
            # self.epoch = latest_state['epoch'] + 1
            # self.i_tb = latest_state['i_tb']
            # self.train_record = latest_state['train_record']
            # self.exp_path = latest_state['exp_path']
            # self.exp_name = latest_state['exp_name']

            latest_state = torch.load(cfg.RESUME_PATH)
            try:
                self.net.load_state_dict(latest_state)
            except:
                self.net.load_state_dict({k.replace('module.', ''): v for k, v in latest_state.items()})

        # self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)

    def forward(self):

        self.test_V1()


    def test_V1(self):  # test_v1 for SHHA, SHHB, UCF-QNRF, UCF50, AC

        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            print(vi)
            img = data[0]
            gt_map = data[1]
            audio_img = data[2]

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                audio_img = Variable(audio_img).cuda()

                if 'Audio' in self.net_name:
                    pred_map = self.net([img, audio_img], gt_map)
                else:
                    pred_map = self.net(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
                # if vi == 0:
                #     vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

                save_img_name = 'val-' + str(vi) + '.jpg'
                raw_img = self.restore_transform(img.data.cpu()[0, :, :, :])
                log_mel = audio_img.data.cpu().numpy()

                raw_img.save(os.path.join(self.save_path, 'raw_img' + save_img_name))
                pyplot.imsave(os.path.join(self.save_path, 'log-mel-map' + save_img_name), log_mel[0, 0, :, :],
                              cmap='jet')

                pred_save_img_name = 'val-' + str(vi) + '-' + str(pred_cnt) + '.jpg'
                gt_save_img_name = 'val-' + str(vi) + '-' + str(gt_count) + '.jpg'
                pyplot.imsave(os.path.join(self.save_path, 'gt-den-map' + '-' + gt_save_img_name), gt_map[0, :, :],
                              cmap='jet')
                pyplot.imsave(os.path.join(self.save_path, 'pred-den-map' + '-' + pred_save_img_name),
                              pred_map[0, 0, :, :],
                              cmap='jet')

        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        # self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        # self.writer.add_scalar('test_mae', mae, self.epoch + 1)
        # self.writer.add_scalar('test_mse', mse, self.epoch + 1)
        print('test_mae: %.5f, test_mse: %.5f, test_loss: %.5f' % (mae, mse, loss))


