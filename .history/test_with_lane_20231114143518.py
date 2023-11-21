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
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, non_max_suppression_seg, \
    mask_iou, masks_iou, process_semantic_mask
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel
from collections import defaultdict
import cv2

from models.lane_model import ENet
import torchvision.transforms as transforms
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

def predict_lane(net, input_img, size):
    net.eval()
    # do the forward pass
    outputs = net(input_img)[-1]
    
    all_cls_pred = torch.zeros((1, 1, 384, 672)).float()
    torch_resize = transforms.Resize((384, 672))
    for cls_idx in range(5):
        if cls_idx in [0, 2, 4]:
            continue
        sig_output = torch.sigmoid(outputs['hm'])[:, cls_idx]
        sig_output = torch_resize(sig_output)

        #print("pred: ", sig_output.shape) # torch.Size([1, 72, 208])
        #print("target: ", input_mask.shape) # torch.Size([1, 1, 72, 208])

        pred_mask = torch.unsqueeze(sig_output.float() , 0)
        
        #print("pred_mask: ", pred_mask.shape) # torch.Size([1, 14976])
        #print("gt_mask: ", gt_mask.shape) # torch.Size([1, 14976])
        all_cls_pred[pred_mask>0.3]=1
    
    all_cls_pred_dilated = all_cls_pred.detach().cpu().numpy()
    all_cls_pred_dilated = torch.from_numpy(ndimage.binary_dilation(all_cls_pred_dilated)).contiguous().float().cuda()
    return all_cls_pred_dilated


def test(data,
         weights=None,
         lane_weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         person_only=False,
         valid_cls_idx=[]):
    # Initialize/load model and set device

    set_logging()
    device = select_device(opt.device, batch_size=batch_size)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    
    if trace:
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

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    
    valid_idx = data.get('valid_idx', None)

    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '), valid_idx=valid_idx, load_seg=True)[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, ap, ap_class, wandb_images = [], [], [], []
    size_stats = None
    stats = []

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

    miou = [[] for _ in range(2)]
    for batch_i, (img, targets, paths, shapes, masks) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        masks = masks.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()            
            out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

            proto = train_out[1]

        #print(out)#[torch.size([bboxnum, 6])*32]
        # Statistics per image
        for si, pred in enumerate(out):
            seg_pred = process_semantic_mask(proto[si], pred[:, 6:], pred[:, :6], img.shape[2:], upsample=True)
            labels = targets[targets[:, 0] == si, 1:]            
            #print(labels) # [[  cls,  x,  y,  w,  h], ... ]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1
            
            lane_img = cv2.imread(str(path.resolve())).astype(np.float32)/255. # (H, W, 3)
            lane_img = cv2.resize(lane_img[14:, :, :], (1664, 576), interpolation=cv2.INTER_LINEAR)
            lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)
            
            lane_img, _ = data_transforms((lane_img, lane_img))
            lane_img = torch.from_numpy(lane_img).permute(2, 0, 1).contiguous().float().unsqueeze(0).to(device)
        
            pred_masks_per_cls = torch.zeros((nc, masks.shape[1], masks.shape[2]), dtype=torch.long, device=device)
            semantic_gt_mask = torch.zeros((nc, masks.shape[1], masks.shape[2]), dtype=torch.long, device=device)
            for t_cls_idx, t_cls in enumerate(targets[targets[:, 0] == si, 1]):
                pred_masks_per_cls[int(t_cls)][seg_pred==t_cls+1] = 1
                semantic_gt_mask[int(t_cls)][((masks[targets[:, 0] == si][t_cls_idx])!=0).bool()] = 1

            all_cls_pred_dilated = predict_lane(model_lane, lane_img, (masks.shape[1], masks.shape[2]))            
            all_cls_pred_dilated_flat = torch.flatten(all_cls_pred_dilated.float(), start_dim=1)

            pred_mask = torch.flatten(pred_masks_per_cls.float(), start_dim=1)
            pred_mask[-1] = all_cls_pred_dilated_flat
            gt_mask = torch.flatten(semantic_gt_mask.float(), start_dim=1)
            ious = torch.squeeze(masks_iou(pred_mask, gt_mask), 0)
            if torch.max(ious) > 0:
                max_ious_idx = torch.argmax(ious)
                if max_ious_idx == 12:
                    miou[0].append(ious[max_ious_idx])
                elif max_ious_idx == 13:
                    miou[1].append(ious[max_ious_idx])

            #vis mask start
            vis_img = cv2.imread(str(path.resolve()))
            image_masks = semantic_gt_mask[-1].detach().cpu().numpy().astype(float)#[label_indexing]
            image_masks = cv2.resize(image_masks, (vis_img.shape[1], vis_img.shape[0]), interpolation = cv2.INTER_NEAREST)
            
            vis_mask = vis_img.copy()
            vis_mask[image_masks!=0] = np.array([50,50,255])
            alpha = 0.5
            vis_img = cv2.addWeighted(vis_img, alpha, vis_mask, 1 - alpha, 0)
            
            image_masks = pred_masks_per_cls[-1].detach().cpu().numpy().astype(float)#[label_indexing]
            image_masks = cv2.resize(image_masks, (vis_img.shape[1], vis_img.shape[0]), interpolation = cv2.INTER_NEAREST)
            
            vis_mask = vis_img.copy()
            vis_mask[image_masks!=0] = np.array([255,50,50])
            alpha = 0.5
            vis_img = cv2.addWeighted(vis_img, alpha, vis_mask, 1 - alpha, 0)

            if torch.max(ious) > 0:
                tl = 2
                vis_txt = str(ious[max_ious_idx])
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(vis_txt, 0, fontScale=tl / 3, thickness=tf)[0]
                c1 = (0, t_size[1]*2)
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(vis_img, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                cv2.putText(vis_img, vis_txt, (c1[0], c1[1] - 2), 0, tl / 3, [255, 200, 255], thickness=tf, lineType=cv2.LINE_AA)
                cv2.imwrite('test_iou/'+str(path.resolve()).split('/')[-1], vis_img)
            #vis mask end

    names = {0: 'drivable area', 1: 'lane'}
    for c, iou in enumerate(miou):
        if len(iou) > 0:
            print(names[c] , " : ", sum(iou)/len(iou))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--lane-weights', type=str, default='yolov7.pt', help='model.pt path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--person-only', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--xyxy', action='store_true', help='the box label type is xyxy not xywh')
    parser.add_argument('--valid-cls-idx', nargs='+', type=int, default=[], help='labels to include when calculating mAP')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.lane_weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             person_only=opt.person_only,
             valid_cls_idx=opt.valid_cls_idx
             )