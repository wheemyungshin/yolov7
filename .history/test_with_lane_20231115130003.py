import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_file, check_img_size, check_requirements, \
    non_max_suppression, increment_path, colorstr, masks_iou, process_semantic_mask
from utils.torch_utils import select_device, TracedModel
import cv2

from models.lane_model import ENet
import torchvision.transforms as transforms
import random

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
        return out_images

def predict_lane(net, input_img, size):
    net.eval()
    # do the forward pass
    outputs = net(input_img)[-1]
    
    all_cls_pred_dilated = np.zeros((1, size[0], size[1]))
    for cls_idx in range(5):
        if cls_idx in [0, 2, 4]:
            continue
        sig_output = torch.sigmoid(outputs['hm'])[:, cls_idx]
        vaf = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
        np_sig_output = sig_output[0].detach().cpu().float().numpy()
        
        np_sig_output = cv2.resize(np_sig_output, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
        vaf = cv2.resize(vaf, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)

        sig_output_mask = np.zeros_like(np_sig_output)
        sig_output_mask[np_sig_output>0.3]=1

        rows, cols = np.nonzero(sig_output_mask)
        for r, c in zip(rows, cols):
            sp = (int(c-vaf[r, c, 0]*3), int(r-vaf[r, c, 1]*3))
            ep = (int(c+vaf[r, c, 0]*3), int(r+vaf[r, c, 1]*3))

            if r > size[0]*4/5:
                thickness=3
            elif r > size[0]*3/4:
                thickness=2
            else:
                thickness=1

            cv2.line(sig_output_mask, sp, ep, 1, thickness)

        sig_output_mask = np.expand_dims(sig_output_mask, axis=0)
        all_cls_pred_dilated[sig_output_mask!=0]=1
    
    all_cls_pred_dilated = torch.from_numpy(all_cls_pred_dilated).contiguous().float().cuda()
    return all_cls_pred_dilated


def test(data,
         weights=None,
         lane_weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_hybrid=False,  # for hybrid auto-labelling
         plots=True,
         half_precision=True,
         is_coco=False):
    # Initialize/load model and set device
    device = select_device(opt.device, batch_size=batch_size)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    
    model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = 'val'  # path to train/val/test images
    
    valid_idx = data.get('valid_idx', None)

    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.0, rect=True,
                                    prefix=colorstr(f'{task}: '), valid_idx=valid_idx, load_seg=True)[0]
    
    #lane detection
    data_mean = [0.485, 0.456, 0.406] #[103.939, 116.779, 123.68]
    data_std = [0.229, 0.224, 0.225] #[1, 1, 1]
    data_transforms = transforms.Compose([
                    GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                    GroupNormalize(mean=(data_mean, (0, )), std=(data_std, (1, ))),
                ])

    heads = {'hm': 5, 'vaf': 2, 'haf': 1}
    model_lane = ENet(heads=heads)

    if lane_weights is not None:
        pretrained_dict = {k: v for k, v in torch.load(lane_weights).items() if k in model_lane.state_dict()}
            #and k not in ['hm.2.weight', 'hm.2.bias', 'haf.2.weight', 'haf.2.bias']}
        model_lane.load_state_dict(pretrained_dict, strict=False)
        
    if device.type != 'cpu':
        model_lane.to(device)

    miou = []
    for batch_i, (img, targets, paths, shapes, masks) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        masks = masks.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model          
            out, train_out = model(img, augment=augment)  # inference and training outputs

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)

            proto = train_out[1]

        # Statistics per image
        for si, pred in enumerate(out):
            seg_pred = process_semantic_mask(proto[si], pred[:, 6:], pred[:, :6], img.shape[2:], upsample=True)
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            
            lane_img = cv2.imread(str(path.resolve())).astype(np.float32)/255. # (H, W, 3)
            lane_img = cv2.resize(lane_img[14:, :, :], (1664, 576), interpolation=cv2.INTER_LINEAR)
            lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
            
            lane_img, _ = data_transforms((lane_img, lane_img))
            lane_img = torch.from_numpy(lane_img).permute(2, 0, 1).contiguous().float().unsqueeze(0).to(device)
        
            pred_masks_per_cls = torch.zeros((1, masks.shape[1], masks.shape[2]), dtype=torch.long, device=device)
            semantic_gt_mask = torch.zeros((1, masks.shape[1], masks.shape[2]), dtype=torch.long, device=device)
            for t_cls_idx, t_cls in enumerate(targets[targets[:, 0] == si, 1]):
                if int(t_cls) >= 12:
                    pred_masks_per_cls[0][seg_pred==t_cls+1] = 1
                    semantic_gt_mask[0][((masks[targets[:, 0] == si][t_cls_idx])!=0).bool()] = 1

            all_cls_pred_dilated = predict_lane(model_lane, lane_img, (masks.shape[1], masks.shape[2]))
            pred_masks_per_cls[all_cls_pred_dilated!=0] = 1      

            pred_mask = torch.flatten(pred_masks_per_cls.float(), start_dim=1)
            gt_mask = torch.flatten(semantic_gt_mask.float(), start_dim=1)
            ious = torch.squeeze(masks_iou(pred_mask, gt_mask), 0)
        
            max_ious_idx = torch.argmax(ious)
            miou.append(ious[max_ious_idx])

    print("drivable area + lane : ", (sum(miou)/len(miou)).item()*100, " %")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--lane-weights', type=str, default='yolov7.pt', help='model.pt path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    test(opt.data,
        opt.weights,
        opt.lane_weights,
        opt.batch_size,
        opt.img_size,
        opt.conf_thres,
        opt.iou_thres,
        opt.single_cls,
        opt.augment,
        opt.verbose,
        save_hybrid=opt.save_hybrid
        )