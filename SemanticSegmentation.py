import os
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from semseg.util import config
from semseg.util.util import colorize
from semseg.model.pspnet import PSPNet
from semseg.model.psanet import PSANet


cv2.ocl.setUseOpenCL(False)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='./semseg/config/cityscapes/cityscapes_psanet101.yaml', help='config file')
    parser.add_argument('--image', type=str, default='/media/bear/T7/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000011517.png', help='input image')
    parser.add_argument('opts', help='see config/cityscapes/cityscapes_psanet101.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.image = args.image
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ['train', 'val', 'test']
    if args.arch == 'psp':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == 'psa':
        if args.compact:
            args.mask_h = (args.train_h - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.train_w - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (args.mask_h is not None and args.mask_w is not None)
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                args.mask_w = 2 * ((args.train_w - 1) // (8 * args.shrink_factor) + 1) - 1
            else:
                assert (args.mask_h % 2 == 1) and (args.mask_h >= 3) and (
                        args.mask_h <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
                assert (args.mask_w % 2 == 1) and (args.mask_w >= 3) and (
                        args.mask_w <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


class SegmentationNetwork:
    def __init__(self, args):
        self.model = None
        if args.arch == 'psp':
            self.model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
        elif args.arch == 'psa':
            self.model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, compact=args.compact,
                           shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                           normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax, pretrained=False)
        self.model = torch.nn.DataParallel(self.model).cuda()

        cudnn.benchmark = True
        args.model_path = os.path.join('./semseg/', args.model_path)
        if os.path.isfile(args.model_path):
            checkpoint = torch.load(args.model_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        self.mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        self.std = [item * value_scale for item in std]
        self.colors = np.loadtxt(os.path.join('./semseg/', args.colors_path)).astype('uint8')
        self.classes = args.classes
        self.scales = args.scales
        self.base_size = args.base_size
        self.crop_h = args.test_h
        self.crop_w = args.test_w
        self.model.eval()

    def test(self, image):
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order

        h, w, _ = image.shape
        prediction = np.zeros((h, w, self.classes), dtype=float)
        for scale in self.scales:
            long_size = round(scale * self.base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)

            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += self.scale_process(image_scale, h, w)
        prediction = self.scale_process(image_scale, h, w)
        prediction = np.argmax(prediction, axis=2)
        gray = np.uint8(prediction)
        color = colorize(gray, self.colors)

        return color
        # image_name = image_path.split('/')[-1].split('.')[0]
        # gray_path = os.path.join('./semseg/figure/demo/', image_name + '_gray.png')
        # color_path = os.path.join('./semseg/figure/demo/', image_name + '_color.png')
        # cv2.imwrite(gray_path, gray)
        # color.save(color_path)

    def net_process(self, image, flip=True):
        input = torch.from_numpy(image.transpose((2, 0, 1))).float()
        if self.std is None:
            for t, m in zip(input, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(input, self.mean, self.std):
                t.sub_(m).div_(s)
        input = input.unsqueeze(0).cuda()
        if flip:
            input = torch.cat([input, input.flip(3)], 0)
        with torch.no_grad():
            output = self.model(input)
        _, _, h_i, w_i = input.shape
        _, _, h_o, w_o = output.shape
        if (h_o != h_i) or (w_o != w_i):
            output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
        output = F.softmax(output, dim=1)
        if flip:
            output = (output[0] + output[1].flip(2)) / 2
        else:
            output = output[0]
        output = output.data.cpu().numpy()
        output = output.transpose(1, 2, 0)
        return output

    def scale_process(self, image, h, w, stride_rate=2 / 3):
        ori_h, ori_w, _ = image.shape
        pad_h = max(self.crop_h - ori_h, 0)
        pad_w = max(self.crop_w - ori_w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.mean)
        new_h, new_w, _ = image.shape
        stride_h = int(np.ceil(self.crop_h * stride_rate))
        stride_w = int(np.ceil(self.crop_w * stride_rate))
        grid_h = int(np.ceil(float(new_h - self.crop_h) / stride_h) + 1)
        grid_w = int(np.ceil(float(new_w - self.crop_w) / stride_w) + 1)
        prediction_crop = np.zeros((new_h, new_w, self.classes), dtype=float)
        count_crop = np.zeros((new_h, new_w), dtype=float)
        for index_h in range(0, grid_h):
            for index_w in range(0, grid_w):
                s_h = index_h * stride_h
                e_h = min(s_h + self.crop_h, new_h)
                s_h = e_h - self.crop_h
                s_w = index_w * stride_w
                e_w = min(s_w + self.crop_w, new_w)
                s_w = e_w - self.crop_w
                image_crop = image[s_h:e_h, s_w:e_w].copy()
                count_crop[s_h:e_h, s_w:e_w] += 1
                prediction_crop[s_h:e_h, s_w:e_w, :] += self.net_process(image_crop)
        prediction_crop /= np.expand_dims(count_crop, 2)
        prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
        prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return prediction

def main():
    args = get_parser()
    check(args)
    semantic_net = SegmentationNetwork(args)

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
    semantic_net.test(image)







if __name__ == '__main__':
    main()
