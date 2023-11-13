import os
import json
from datetime import datetime
from statistics import mean
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.lane_model import ENet
import torchvision.transforms as transforms
import cv2
import random

from scipy import ndimage


class GroupRandomScale(object):
    def __init__(self, size=(0.5, 1.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        assert (len(self.interpolation) == len(img_group))
        scale = random.uniform(self.size[0], self.size[1])
        out_images = list()
        for img, interpolation in zip(img_group, self.interpolation):
            out_images.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation))
            if len(img.shape) > len(out_images[-1].shape):
                out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
        return out_images

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img_group):
        out_images = list()
        for img, m, s in zip(img_group, self.mean, self.std):
            if len(m) == 1:
                img = img - np.array(m)  # single channel image
                img = img / np.array(s)
            else:
                img = img - np.array(m)[np.newaxis, np.newaxis, ...]
                img = img / np.array(s)[np.newaxis, np.newaxis, ...]
            out_images.append(img)

        # cv2.imshow('img', (out_images[0] + np.array(self.mean[0])[np.newaxis, np.newaxis, ...]).astype(np.uint8))
        # cv2.imshow('label', (out_images[1] * 100).astype(np.uint8))
        # print(np.unique(out_images[1]))
        # cv2.waitKey()
        return out_images


def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = np.expand_dims(polygons, axis=0)
    cv2.fillPoly(mask, [polygons], color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask

def mask_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [M, n] m2 means number of gt objects
    Note: n means image_w x image_h

    return: masks iou, [N, M]
    """
    intersection = torch.matmul(mask1, mask2.t()).clamp(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)

# validation function
def val(net, sample, args, epoch):
    net.eval()
    
    input_img, input_mask = sample
    if args.cuda:
        input_img = input_img.cuda()
        input_mask = input_mask.cuda()

    # do the forward pass
    outputs = net(input_img)[-1]
    
    all_cls_pred = torch.zeros_like(input_mask.float())
    all_cls_target = torch.zeros_like(torch.flatten(input_mask.float(), start_dim=1))
    for cls_idx in range(args.cls_num):
        if cls_idx in [0, 2, 4]:
            continue
        sig_output = torch.sigmoid(outputs['hm'])[:, cls_idx]
        pred = sig_output.detach().cpu().numpy().ravel()
        target = input_mask.detach().cpu().numpy().ravel()

        #print("pred: ", sig_output.shape) # torch.Size([1, 72, 208])
        #print("target: ", input_mask.shape) # torch.Size([1, 1, 72, 208])

        pred_mask = torch.unsqueeze(sig_output.float() , 0)
        
        #print("pred_mask: ", pred_mask.shape) # torch.Size([1, 14976])
        #print("gt_mask: ", gt_mask.shape) # torch.Size([1, 14976])
        all_cls_pred[pred_mask>0.3]=1
    
    all_cls_pred_dilated = all_cls_pred.detach().cpu().numpy()
    all_cls_pred_dilated = torch.from_numpy(ndimage.binary_dilation(all_cls_pred_dilated)).contiguous().float().cuda()
    all_cls_pred_dilated = torch.flatten(all_cls_pred_dilated.float(), start_dim=1)

    gt_mask = torch.flatten(input_mask.float(), start_dim=1)
    all_cls_target[gt_mask>0.5]=1
    if torch.sum(all_cls_target) > 0:
        ious = torch.squeeze(mask_iou(all_cls_pred_dilated, all_cls_target), 0)
        max_ious_idx = torch.argmax(ious)

    return max_ious_idx, ious[max_ious_idx].detach().cpu().numpy()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Options for training LaneAF models in PyTorch...')
    parser.add_argument('--dataset-dir', type=str, default=None, help='path to dataset')
    parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
    parser.add_argument('--epochs', type=int, default=40, metavar='N', help='number of epochs to train for')
    parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, metavar='WD', help='weight decay')
    parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of iterations to print/save log after')
    parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
    parser.add_argument('--cls-num', type=int, default=1, help='class num to predict')
    args = parser.parse_args()
    

    # check args
    if args.dataset_dir is None:
        assert False, 'Path to dataset not provided!'

    # setup args
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    data_mean = [0.485, 0.456, 0.406] #[103.939, 116.779, 123.68]
    data_std = [0.229, 0.224, 0.225] #[1, 1, 1]
    data_transforms = transforms.Compose([
                    GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                    GroupNormalize(mean=(data_mean, (0, )), std=(data_std, (1, ))),
                ])

    print("Data loading...")
    val_loader = []
    with open(os.path.join(args.dataset_dir, 'test.txt'), 'r') as f:
        image_path_lines = f.readlines()
        for image_path_line in image_path_lines:
            image_path_line = image_path_line.strip()
            image_path = os.path.join(args.dataset_dir, 'images', image_path_line.split('/images/')[-1])
            seg_path = os.path.join(args.dataset_dir, 'segments', image_path_line.split('/images/')[-1].replace('.jpg', '.txt'))

            img = cv2.imread(image_path).astype(np.float32)/255. # (H, W, 3)
            img = cv2.resize(img[14:, :, :], (1664, 576), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img, _ = data_transforms((img, img))
            height, width, _ = img.shape
            total_mask = np.zeros((int(height/4), int(width/4)))
            # convert all outputs to torch tensors
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().unsqueeze(0)

            with open(seg_path, 'r') as f:
                polygon_lines = f.readlines()
                for polygon_line in polygon_lines:
                    polygon_line_elem = polygon_line.split(' ')
                    polygon_cls = polygon_line_elem[0]
                    polygon_np = []
                    if int(polygon_cls) == 16:
                        for polygon_elem_idx in range(0, len(polygon_line_elem)-2, 2):
                            x = float(polygon_line_elem[polygon_elem_idx+1]) * int(width/4)
                            y = float(polygon_line_elem[polygon_elem_idx+2]) * int(height/4)
                            polygon_np.append([x, y])
                        polygon_np = np.array(polygon_np)
                        
                        mask = polygon2mask((int(height/4), int(width/4)), polygon_np)
                        total_mask[mask!=0] = 1
            
            total_mask_torch = torch.from_numpy(total_mask).contiguous().float().unsqueeze(0).unsqueeze(0)

            val_loader.append((img, total_mask_torch))

    print("Data loading complete.")

    heads = {'hm': args.cls_num, 'vaf': 2, 'haf': 1}
    model = ENet(heads=heads)

    if args.snapshot is not None:
        pretrained_dict = {k: v for k, v in torch.load(args.snapshot).items() if k in model.state_dict()
            and k not in ['hm.2.weight', 'hm.2.bias', 'haf.2.weight', 'haf.2.bias']}
        model.load_state_dict(pretrained_dict, strict=False)
        
    if args.cuda:
        model.cuda()
    print(model)    

    epoch_acc, epoch_f1 = list(), list()    
    miou = [[] for _ in range(args.cls_num)]
    for sample in val_loader:
        val_acc, val_f1 = val(model, sample, args, 0)
        epoch_acc.append(val_acc)
        epoch_f1.append(val_f1)
    
    # now that the epoch is completed calculate statistics and store logs
    avg_acc = mean(epoch_acc)
    avg_f1 = mean(epoch_f1)
    print("\n------------------------ Validation metrics ------------------------")
    print("Average accuracy for epoch = {:.4f}".format(avg_acc))
    print("Average F1 score for epoch = {:.4f}".format(avg_f1))
    print("--------------------------------------------------------------------\n")
    
    for c, iou in enumerate(miou):
        if len(iou) > 0:
            print(str(c) , " : ", sum(iou)/len(iou))