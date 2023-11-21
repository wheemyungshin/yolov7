import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, non_max_suppression_seg, process_mask, scale_masks, process_semantic_mask
from utils.plots import plot_one_box, plot_masks
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import os
import numpy as np
from collections import defaultdict

from models.lane_model import ENet
import torchvision.transforms as transforms
import random

from torchvision.models import resnet101
from PIL import Image
import math

CAMERA_LENS_ANGLE = 90
CAMERA_ANGLE_WEIGHT = math.tan((CAMERA_LENS_ANGLE/2/180)*math.pi)
OVERALL_HEIGHT_DICT = { # meter
    2 : 0.4,
    4 : 1.6, # 길지 않아서 왜곡 정도 낮음, 그러나 자세 등으로 오히려 작아질 가능성 큼
    6 : 1.8, # 차량류는 각도 등에 의해 박스 크기는 실제 전고보다 길 것을 반영하여 조금 더 크게 잡는다
    7 : 2.5, # 가장 차이 심함
    8 : 3.5,
    10 : 1.7,
    11 : 1.7
}


def calculate_distance(cxywh):
    c = cxywh[0]
    x = cxywh[1]
    y = cxywh[2]
    w = cxywh[3]
    h = cxywh[4]

    #distance = 191.42 * math.exp(-13.35 * h) # 예시 샘플링
    #distance = 2.5 * 1/h # 거리로부터 전고 계산 25m : 10% 기준의 단순 평면 FOV (전고 2.17m)
    #distance = ((2.5 * 1/h) + (191.42 * math.exp(-13.35 * h))) / 2 #반반


    #distance = 2 * 1/h # 전고로부터 거리 계산 (역순) 20.7m : 10% 기준의 단순 평면 FOV (승용차 전고 1.8m)
    benchmark_distance = (OVERALL_HEIGHT_DICT[c]) * (10/2) / CAMERA_ANGLE_WEIGHT # 화면상 10%를 차지할 때의 거리
    distance = (benchmark_distance/10) * 1/h

    distance = round(distance, 3)

    return distance

# Lane 데이터 Resize 전처리 정의
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

# Lane 데이터 Normalize 전처리 정의
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


def detect(save_img=False):
    warn_status = {
        'last_light' : ['No light', 0],
        'closest_pedestrian' : -1,
        'on_road' : False,
        'on_side' : None,
        'crossline' : [False, 0],
    }
    lane_x_medians = [None, None, None, None, None, None]
    bbox_num = 0
    bbox_num_per_cls = defaultdict(int)
    
    source, weights, weights_seg, weights_lane, weights_cls, view_img, save_txt, imgsz = opt.source, opt.weights, opt.weights_seg, opt.weights_lane, opt.weights_cls, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if len(opt.img_size)==2:
        model = TracedModel(model, device, tuple(opt.img_size))
    elif len(opt.img_size)==1:
        model = TracedModel(model, device, tuple([opt.img_size[0], opt.img_size[0]]))
    

    model_seg = attempt_load(weights_seg, map_location=device)  # load FP32 model
    if len(opt.img_size)==2:
        model_seg = TracedModel(model_seg, device, tuple(opt.img_size))
    elif len(opt.img_size)==1:
        model_seg = TracedModel(model_seg, device, tuple([opt.img_size[0], opt.img_size[0]]))


    # Lane detection 모델 불러오기
    heads = {'hm': 5, 'vaf': 2, 'haf': 1}
    model_lane = ENet(heads=heads)
    if weights_lane is not None:
        pretrained_dict = {k: v for k, v in torch.load(weights_lane).items() if k in model_lane.state_dict()}
            #and k not in ['hm.2.weight', 'hm.2.bias', 'haf.2.weight', 'haf.2.bias']}
        model_lane.load_state_dict(pretrained_dict, strict=False)        
    if device.type != 'cpu':
        model_lane.to(device)
    model_lane.eval()

    
    # 신호등 classification 모델 불러오기
    model_cls = resnet101(pretrained=True)
    model_cls.fc = nn.Linear(2048, 3)
    model_cls.load_state_dict(torch.load(weights_cls))
    model_cls = model_cls.to('cuda')
    model_cls.eval()

    # cls 모델 전처리 정의
    transform_cls = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    # Lane detection 데이터 전처리 정의
    data_mean = [0.485, 0.456, 0.406] #[103.939, 116.779, 123.68]
    data_std = [0.229, 0.224, 0.225] #[1, 1, 1]
    data_transforms = transforms.Compose([
                    GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                    GroupNormalize(mean=(data_mean, (0, )), std=(data_std, (1, ))),
                ])


    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    road_indices = [13]
    seg_colors = [[255, 100, 100]]#[[random.randint(0, 255) for _ in range(3)] for _ in range(len(road_indices))]
    lane_colors = [[255, 100, 255], [250, 250, 250], [100, 255, 100], [100, 255, 255], [100, 100, 255]]#[[random.randint(0, 255) for _ in range(3)] for _ in range(5)]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
        model_seg(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = imgsz[1]
    old_img_h = imgsz[0]
    old_img_b = 1

    if opt.save_frame:
        os.makedirs(os.path.join(save_dir, 'vis_frames'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'mask_npy'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'mask_npy', 'lane'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'mask_npy', 'drivable_area'), exist_ok=True)

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if opt.frame_ratio > 0:
            frame_ratio = opt.frame_ratio
        else:
            fps = 30

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]
                model_seg(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred_seg, out = model_seg(img, augment=opt.augment)
        proto = out[1]
        
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred_seg = non_max_suppression(pred_seg, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)        
        t3 = time_synchronized()

        # Process detections
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        clean_im0 = im0.copy()

        if opt.frame_ratio <= 0:
            frame_ratio = fps
            
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        gn_to = torch.tensor(img.shape)[[3, 2, 3, 2]]
                    

        if dataset.mode != 'image' and frame % frame_ratio != 0:
            continue
        else:
            # Lane 데이터 전처리
            lane_img = clean_im0.copy().astype(np.float32)/255. # (H, W, 3)
            lane_img = cv2.resize(lane_img[14:, :, :], (1664, 576), interpolation=cv2.INTER_LINEAR)
            lane_img = cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB)            
            lane_img, _ = data_transforms((lane_img, lane_img))
            lane_img = torch.from_numpy(lane_img).permute(2, 0, 1).contiguous().float().unsqueeze(0).to(device)        

            # Lane 모델 돌리기
            outputs = model_lane(lane_img)[-1]
            
            # Lane 예측 결과 후처리 (선 그어주기)
            save_lane_np = np.zeros((5, 96, 160))
            all_cls_pred_dilated = np.zeros((5, imgsz[0], imgsz[1]))
            for cls_idx in range(5):
                sig_output = torch.sigmoid(outputs['hm'])[:, cls_idx] #좌표
                vaf = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0)) #방향
                np_sig_output = sig_output[0].detach().cpu().float().numpy()                
                vaf = cv2.resize(vaf, (imgsz[1], imgsz[0]), interpolation=cv2.INTER_CUBIC)
                np_sig_output = cv2.resize(np_sig_output, (imgsz[1], imgsz[0]), interpolation=cv2.INTER_CUBIC)
                sig_output_mask = np.zeros_like(np_sig_output)
                sig_output_mask[np_sig_output>0.3]=1
                rows, cols = np.nonzero(sig_output_mask)
                # 선 그어주기
                for r, c in zip(rows, cols):
                    sp = (int(c-vaf[r, c, 0]*3), int(r-vaf[r, c, 1]*3))
                    ep = (int(c+vaf[r, c, 0]*3), int(r+vaf[r, c, 1]*3))
                    if r > imgsz[0]*4/5:
                        thickness=3
                    elif r > imgsz[0]*3/4:
                        thickness=2
                    else:
                        thickness=1
                    cv2.line(sig_output_mask, sp, ep, 1, thickness)
                all_cls_pred_dilated[cls_idx][sig_output_mask!=0]=1

                save_lane_np_ = all_cls_pred_dilated[cls_idx].copy()
                save_lane_np[cls_idx] = cv2.resize(save_lane_np_, (160, 96), interpolation = cv2.INTER_NEAREST)
            np.save(os.path.join(save_dir, 'mask_npy', 'lane', p.name.split('.')[0]+'_'+str(frame)), save_lane_np)

            lane_x_median = None
            for cls_idx in range(5):
                cls_pred_dilated = all_cls_pred_dilated[cls_idx]
                cls_pred_dilated = cv2.resize(cls_pred_dilated, (im0.shape[1], im0.shape[0]), interpolation = cv2.INTER_NEAREST)

                vis_mask = im0.copy()
                vis_mask[cls_pred_dilated!=0] = np.array(lane_colors[cls_idx])
                alpha = 0.5
                im0 = cv2.addWeighted(im0, alpha, vis_mask, 1 - alpha, 0)

                if cls_idx == 3: # yellow line
                    lane_roi = cls_pred_dilated[int(image_masks.shape[0]*2/3):, int(image_masks.shape[1]*2/5):int(image_masks.shape[1]*3/5)]
                    lane_xs, lane_ys = np.nonzero(lane_roi)
                    if len(lane_xs) > 0:
                        lane_x_median = int(np.median(lane_xs))
                        warn_status['on_side'] = lane_x_median
            
            lane_x_medians[frame % len(lane_x_medians)] = lane_x_median
            lane_med_sum_prev = 0
            lane_med_sum_temp = 0
            lane_med_num_prev = 0
            lane_med_num_temp = 0
            for lane_med_idx, lane_med in enumerate(lane_x_medians):
                if lane_med is not None:
                    if lane_med_idx < int(len(lane_x_medians)/2):
                        lane_med_sum_prev+=lane_med
                        lane_med_num_prev+=1
                    else:
                        lane_med_sum_temp+=lane_med
                        lane_med_num_temp+=1
            lane_med_sum_prev /= lane_med_num_prev
            lane_med_sum_temp /= lane_med_num_temp
            if lane_med_sum_prev < int(image_masks.shape[1]*5/11) and int(image_masks.shape[1]*5/11) < lane_med_sum_temp:



            warn_status['on_road'] = False
            for i, det in enumerate(pred_seg):  # detections per image
                if len(det):
                    masks = process_semantic_mask(proto[i], det[:, 6:], det[:, :6], img.shape[2:], upsample=True)
                
                    #for semantic masks
                    image_masks = masks.detach().cpu().numpy().astype(float)#[label_indexing]
                    image_masks = cv2.resize(image_masks, (im0.shape[1], im0.shape[0]), interpolation = cv2.INTER_NEAREST)

                    save_road_np = image_masks.copy()
                    save_road_np = cv2.resize(save_road_np, (160, 96), interpolation = cv2.INTER_NEAREST)

                    np.save(os.path.join(save_dir, 'mask_npy', 'drivable_area', p.name.split('.')[0]+'_'+str(frame)), save_road_np)
                    
                    vis_mask = im0.copy()
                    for road_idx, image_mask_idx in enumerate(road_indices):
                        vis_mask[image_masks==image_mask_idx] = np.array(seg_colors[road_idx])
                    alpha = 0.5
                    im0 = cv2.addWeighted(im0, alpha, vis_mask, 1 - alpha, 0)

                    on_road_roi = image_masks[int(image_masks.shape[0]/2):, int(image_masks.shape[1]/3):int(image_masks.shape[1]*2/3)]
                    on_road_ratio = np.sum(on_road_roi==13) / (on_road_roi.shape[0]*on_road_roi.shape[1])
                    if on_road_ratio > 0.3:
                        warn_status['on_road'] = True
                    '''
                    left_side_road = image_masks[:, :lane_x_median]
                    right_side_road = image_masks[:, lane_x_median:]
                    if np.sum(left_side_road==13)*3 < np.sum(right_side_road==13):
                        if warn_status['on_side'] == 'left':
                            warn_status['crossline'] = [True, 30]
                        warn_status['on_side'] = 'right'
                    elif np.sum(left_side_road==13) > np.sum(right_side_road==13)*3:
                        if warn_status['on_side'] == 'right':
                            warn_status['crossline'] = [True, 30]
                        warn_status['on_side'] = 'left'
                    else:
                        warn_status['on_side'] = 'None'
                    '''



            max_traffic_light_size = 32 # (1080 x 1920)
            max_person_size = 0 # (1080 x 1920)
            closest_pedestrian_distance = -1
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    scores = det[:, 4]
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    # Write results
                    for *xyxy, conf, cls in reversed(det[:, :6]):
                        if cls in [2, 4, 6, 7, 8, 10, 11]:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            if save_txt:  # Write to file                                
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                distance_str = str(calculate_distance([int(cls), float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])])) + 'm'
                                label = f'{names[int(cls)]} {conf:.2f} {distance_str}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                            
                            bbox_num+=1
                            bbox_num_per_cls[names[int(cls)]]+=1
                        
                            if cls == 2 and conf > 0.5: # traffic light
                                if xyxy[3] - xyxy[1] > max_traffic_light_size and xyxy[2] - xyxy[0] > xyxy[3] - xyxy[1]:
                                    max_traffic_light_size = xyxy[3] - xyxy[1]
                                    margin = 16
                                    crop_x1 = int(xyxy[0]-margin)
                                    crop_y1 = int(xyxy[1]-margin)
                                    crop_x2 = int(xyxy[2]+margin)
                                    crop_y2 = int(xyxy[3]+margin)
                                    if crop_x1 > 0 and crop_y1 > 0 and crop_x2 < clean_im0.shape[1] and crop_y2 < clean_im0.shape[0]:
                                        traffic_light_crop = Image.fromarray(clean_im0.copy()[crop_y1:crop_y2, crop_x1:crop_x2])
                                        traffic_light_crop = torch.unsqueeze(transform_cls(traffic_light_crop), 0).to('cuda')
                                        cls_outputs = model_cls(traffic_light_crop)
                                        _, predicted = torch.max(cls_outputs.data, 1)
                                        if predicted[0] == 0:
                                            warn_status['last_light'] = ['Red', 0]
                                            with open(txt_path + '.txt', 'a') as f:
                                                f.write('Red\n')
                                            print(txt_path)
                                        elif predicted[0] == 1:
                                            warn_status['last_light'] = ['Arrow', 0]
                                            with open(txt_path + '.txt', 'a') as f:
                                                f.write('Arrow\n')
                                            print(txt_path)
                                        elif predicted[0] == 2:
                                            warn_status['last_light'] = ['Green', 0]
                                            with open(txt_path + '.txt', 'a') as f:
                                                f.write('Green\n')
                                            print(txt_path)
                                        
                            
                            if cls == 4: # person
                                if xyxy[3] - xyxy[1] > max_person_size:
                                    max_person_size = xyxy[3] - xyxy[1]
                                    closest_pedestrian_distance = calculate_distance([int(cls), float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])])
                                    warn_status['closest_pedestrian'] = closest_pedestrian_distance


                if warn_status['last_light'][0] != 'No light':
                    warn_status['last_light'][1] += 1
                if warn_status['last_light'][1] > 300 : 
                    warn_status['last_light'][0] = 'No light'
                    warn_status['last_light'][1] = 0

                if warn_status['crossline'][0]:
                    warn_status['crossline'][1] -= 1
                    if warn_status['crossline'][1] <= 0: 
                        warn_status['crossline'][0] = False
                        warn_status['crossline'][1] = 0

                tl = 3
                warn_key_idx = 0
                for warn_key in warn_status.keys():
                    if warn_key == 'last_light':
                        if warn_status[warn_key][0] == 'No light':
                            print_value = warn_key + ' : ' + warn_status[warn_key][0]
                            print_color = [255, 255, 255]
                        else:
                            print_value = warn_key + ' : ' + warn_status[warn_key][0]
                            if warn_status[warn_key][0] == 'Red':
                                print_color = [0, 0, 255]
                            elif warn_status[warn_key][0] == 'Arrow':
                                print_color = [255, 100, 255]
                            elif warn_status[warn_key][0] == 'Green':
                                print_color = [127, 255, 0]
                            else:
                                print_color = [0, 0, 0]
                    
                    if warn_key == 'closest_pedestrian':
                        #if warn_status[warn_key] == -1:
                        #    print_value = warn_key + ' : ' + 'No close person'
                        #    print_color = [255, 255, 255]
                        #else:
                        #    print_value = warn_key + ' : ' + str(warn_status[warn_key])
                        #    print_color = [0, 0, 255]
                        print_value = warn_key + ' : ' + str(warn_status[warn_key]) + ' m'
                        if warn_status[warn_key] == -1:
                            print_value = warn_key + ' : ' + 'No person'
                            print_color = [255, 255, 255]
                        elif warn_status[warn_key] > 2:
                            print_color = [255, 255, 255]
                        else:
                            print_color = [0, 0, 255]

                    if warn_key == 'on_road':
                        if warn_status[warn_key]:
                            print_value = warn_key + ' : ' + 'On'
                            print_color = [255, 255, 255]
                        else:
                            print_value = warn_key + ' : ' + 'Off'
                            print_color = [0, 0, 255]
                    
                    if warn_key == 'on_side':
                        if warn_status[warn_key] is not None:
                            print_value = warn_key + ' : ' + str(lane_x_medians)# + warn_status[warn_key]
                            print_color = [0, 255, 255]
                        else:
                            print_value = warn_key + ' : ' + str(lane_x_medians)# + 'None'
                            print_color = [255, 255, 255]
                        
                    if warn_key == 'crossline':
                        if warn_status[warn_key][0]:
                            print_value = warn_key + ' : ' + 'Yes'
                            print_color = [0, 0, 255]
                        else:
                            print_value = warn_key + ' : ' + 'No'
                            print_color = [255, 255, 255]


                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(print_value, 0, fontScale=tl / 3, thickness=tf)[0]
                    c1 = (0, int(t_size[1]*2*warn_key_idx+3))
                    c2 = c1[0] + t_size[0], c1[1] + t_size[1]*2
                    cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                    cv2.putText(im0, print_value, (c1[0], c2[1] - int(t_size[1]/2)), 0, tl / 3, print_color, thickness=tf, lineType=cv2.LINE_AA)

                    warn_key_idx+=1


                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        #cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                        if opt.save_frame:
                            print(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0]))
                            if len(det) > 0:
                                cv2.imwrite(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0])+'_'+str(frame)+'.jpg', im0)
                                cv2.imwrite(os.path.join(save_dir, 'images', p.name.split('.')[0])+'_'+str(frame)+'_clean.jpg', clean_im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        if opt.save_frame:
                            print(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0]))
                            cv2.imwrite(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0])+'_'+str(frame)+'.jpg', im0)
                            if len(det) > 0:
                                cv2.imwrite(os.path.join(save_dir, 'images', p.name.split('.')[0])+'_'+str(frame)+'_clean.jpg', clean_im0)
                        vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    print("BBOX NUM: ", bbox_num)
    for k, v in bbox_num_per_cls.items():
        print(k, " : ", v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--weights-seg', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--weights-lane', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--weights-cls', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder
    parser.add_argument('--img-size', nargs='+', type=int, default=[192,256], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--frame-ratio', default=1, type=int, help='save frame ratio')
    parser.add_argument('--save-frame', action='store_true', help='save each frame of video results')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
