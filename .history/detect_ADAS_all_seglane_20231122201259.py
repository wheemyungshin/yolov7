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

import torchvision.transforms as transforms
import random

from torchvision.models import resnet101
from PIL import Image
import math

def separate_instances(mask):
    mask = np.asarray(mask, dtype="uint8")
    # Find contours of each instance in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list of masks for each instance
    instance_masks = []
    for contour in contours:
        # Create a mask for the current instance
        instance_mask = np.zeros_like(mask)
        cv2.drawContours(instance_mask, [contour], 0, 1, -1)
        instance_masks.append(instance_mask)

    return instance_masks

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
    bbox_num = 0
    bbox_num_per_cls = defaultdict(int)
    
    source, weights, weights_seg, weights_cls, view_img, save_txt, imgsz = opt.source, opt.weights, opt.weights_seg, opt.weights_cls, opt.view_img, opt.save_txt, opt.img_size
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
    road_indices = [13, 14, 15]
    seg_colors = [[255, 100, 100], [100, 255, 255], [255, 100, 255]]#[[random.randint(0, 255) for _ in range(3)] for _ in range(len(road_indices))]

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

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t0_each = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if opt.frame_ratio > 0:
            frame_ratio = opt.frame_ratio
        else:
            fps = 30

        # Inference
        t1 = time_synchronized()
        pred_seg, out = model_seg(img, augment=opt.augment)
        proto = out[1]
        
        t2 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

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

        if vid_path != save_path:  # new video
            last_lights = ['Nolight']*120
            on_roads = [True]*90
            closest_pedestrians = [False]*30
            warn_status = {
                #for voting
                'last_light' : 'Nolight',
                'on_road' : False,
                'closest_pedestrian' : -1,
                'on_left' : None,
                'going_left' : None,
                'speed' : 0,

                #final
                'ignore_light' : [False, 0],
                'crossline' : [False, 0],
                'on_sidewalk' : [False, 0],
                'ignore_pedestrain' : [False, 0],
            }
            #lane_x_medians = [None]*6
            lane_on_lefts = [None]*150
            lane_going_lefts = [None]*150
            prev_gray = None

        if dataset.mode != 'image' and frame % frame_ratio != 0:
            continue
        else:
            on_road = False
            closest_pedestrian = False
            for i, det in enumerate(pred_seg):  # detections per image
                if len(det):
                    masks = process_semantic_mask(proto[i], det[:, 6:], det[:, :6], img.shape[2:], upsample=True)
                
                    #for semantic masks
                    image_masks = masks.detach().cpu().numpy().astype(float)#[label_indexing]
                    image_masks = cv2.resize(image_masks, (im0.shape[1], im0.shape[0]), interpolation = cv2.INTER_NEAREST)

                    save_road_np = image_masks.copy()
                    #save_road_np = cv2.resize(save_road_np, (160, 96), interpolation = cv2.INTER_NEAREST)

                    np.save(os.path.join(save_dir, 'mask_npy', p.name.split('.')[0]+'_'+'0'*(6-len(str(frame)))+str(frame)), save_road_np)
                    
                    vis_mask = im0.copy()
                    for road_idx, image_mask_idx in enumerate(road_indices):
                        vis_mask[image_masks==image_mask_idx] = np.array(seg_colors[road_idx])
                    alpha = 0.5
                    im0 = cv2.addWeighted(im0, alpha, vis_mask, 1 - alpha, 0)

                    on_road_roi = image_masks[int(image_masks.shape[0]/2):, int(image_masks.shape[1]/3):int(image_masks.shape[1]*2/3)]
                    on_road_ratio = np.sum(on_road_roi!=0) / (on_road_roi.shape[0]*on_road_roi.shape[1])
                    if on_road_ratio > 0.3:
                        on_road = True

            lane_masks = separate_instances(image_masks==14)
            largest_area = 0
            largest_mask = None
            for lane_mask in lane_masks:
                if np.sum(lane_mask) > largest_area:
                    largest_mask = lane_mask
            
            if largest_mask is not None:
                lane_ys, lane_xs = np.nonzero(largest_mask)
                lane_ys = largest_mask.shape[0] - lane_ys
                a, b = np.polyfit(lane_xs, lane_ys, 1)
                if a < 0:
                    on_left = True
                elif a > 0:
                    on_left = False
                else:
                    on_left = None
                if on_left is not None:
                    x_intercept = -b/a
                    if x_intercept > largest_mask.shape[1]/2:
                        going_left = True
                    else:
                        going_left = False
            else:
                on_left = None
                going_left = None

            for ongoing_left_idx in range(len(lane_on_lefts)-1):
                lane_on_lefts[ongoing_left_idx] = lane_on_lefts[ongoing_left_idx+1]
                lane_going_lefts[ongoing_left_idx] = lane_going_lefts[ongoing_left_idx+1]
            lane_on_lefts[-1] = on_left
            lane_going_lefts[-1] = going_left

            if sum(lane_going_lefts[:int(len(lane_going_lefts)/2)]) > len(lane_going_lefts)*0.3:
                warn_status['crossline']

            '''
            lane_x_median = None
            lane_roi = (image_masks==14)[int(im0.shape[0]*4/5):, int(im0.shape[1]*2/5):int(im0.shape[1]*3/5)]
            lane_ys, lane_xs = np.nonzero(lane_roi)
            if len(lane_xs) > 0:
                lane_x_median = int(np.median(lane_xs))
            
            for lane_med_idx in range(len(lane_x_medians)-1):
                lane_x_medians[lane_med_idx] = lane_x_medians[lane_med_idx+1]
            lane_x_medians[-1] = lane_x_median
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
            if lane_med_num_prev > 0 and lane_med_num_temp > 0:
                lane_med_sum_prev /= lane_med_num_prev
                lane_med_sum_temp /= lane_med_num_temp
                if lane_med_sum_prev < int((im0.shape[1]/5)*4/9) and int((im0.shape[1]/5)*5/9) < lane_med_sum_temp:
                    warn_status['crossline'] = [True, 30]
                elif lane_med_sum_temp < int((im0.shape[1]/5)*4/9) and int((im0.shape[1]/5)*5/9) < lane_med_sum_prev:
                    warn_status['crossline'] = [True, 30]
            '''
                    
            
            if opt.calc_optical_flow:
                gray = clean_im0.copy()
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    #seg_map = np.zeros_like(image_masks, dtype=np.float64)
                    #seg_map[image_masks!=0]=1
                    
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,  
                                                None, 
                                                0.5, 3, 15, 3, 5, 1.2, 0) 
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    angle_converted = angle / np.pi / 2
                    global_value = magnitude

                    #global_value = global_value * seg_map
                    warn_status['speed'] = np.mean(global_value)

                prev_gray = gray 
            
            last_light = 'Nolight'
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
                        
                            if cls == 2 and conf > 0.3: # traffic light
                                if xyxy[2] - xyxy[0] > max_traffic_light_size and xyxy[2] - xyxy[0] > (xyxy[3] - xyxy[1])*0.9:
                                    max_traffic_light_size = xyxy[2] - xyxy[0]
                                    margin = 32
                                    crop_x1 = int(xyxy[0]-margin)
                                    crop_y1 = int(xyxy[1]-margin)
                                    crop_x2 = int(xyxy[2]+margin)
                                    crop_y2 = int(xyxy[3]+margin)
                                    if crop_x1 > 0 and crop_y1 > 0 and crop_x2 < clean_im0.shape[1] and crop_y2 < clean_im0.shape[0]:
                                        traffic_light_crop = Image.fromarray(clean_im0.copy()[crop_y1:crop_y2, crop_x1:crop_x2, ::-1])
                                        traffic_light_crop = torch.unsqueeze(transform_cls(traffic_light_crop), 0).to('cuda')
                                        cls_outputs = model_cls(traffic_light_crop)
                                        _, predicted = torch.max(cls_outputs.data, 1)
                                        if predicted[0] == 0:
                                            last_light = 'Red'
                                            with open(txt_path + '.txt', 'a') as f:
                                                f.write('Red\n')
                                        elif predicted[0] == 1:
                                            last_light = 'Arrow'
                                            with open(txt_path + '.txt', 'a') as f:
                                                f.write('Arrow\n')
                                        elif predicted[0] == 2:
                                            last_light = 'Green'
                                            with open(txt_path + '.txt', 'a') as f:
                                                f.write('Green\n')
                            if cls == 4: # person
                                if xyxy[3] - xyxy[1] > max_person_size:
                                    max_person_size = xyxy[3] - xyxy[1]
                                    closest_pedestrian_distance = calculate_distance([int(cls), float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])])
                                    
                                    if closest_pedestrian_distance == -1:
                                        closest_pedestrian = False
                                    elif closest_pedestrian_distance > 2:
                                        closest_pedestrian = False
                                    else:
                                        closest_pedestrian = True
                        
                for lane_med_idx in range(len(last_lights)-1):
                    last_lights[lane_med_idx] = last_lights[lane_med_idx+1]
                last_lights[-1] = last_light

                for on_road_idx in range(len(on_roads)-1):
                    on_roads[on_road_idx] = on_roads[on_road_idx+1]
                on_roads[-1] = on_road

                for ped_idx in range(len(closest_pedestrians)-1):
                    closest_pedestrians[ped_idx] = closest_pedestrians[ped_idx+1]
                closest_pedestrians[-1] = closest_pedestrian  

                tl = 3
                warn_key_idx = 0
                for warn_key in warn_status.keys():
                    if warn_key == 'last_light':
                        count_red = last_lights.count('Red')
                        count_arrow = last_lights.count('Arrow')
                        count_green = last_lights.count('Green')
                        if last_lights.count('Nolight') == len(last_lights):
                            warn_status['last_light'] = 'Nolight'
                            print_color = [255, 255, 255]
                        else:
                            if count_red > count_arrow and count_red > count_green:
                                warn_status['last_light'] = 'Red'
                                print_color = [0, 0, 255]
                            elif count_green > count_red and count_green > count_arrow:
                                warn_status['last_light'] = 'Arrow'
                                print_color = [255, 100, 255]
                            elif count_arrow > count_red and count_arrow > count_green:
                                warn_status['last_light'] = 'Green'
                                print_color = [127, 255, 0]
                            else:
                                warn_status['last_light'] = 'Nolight'
                                print_color = [255, 255, 255]
                        print_value = warn_key + ' : ' + warn_status['last_light']
                    
                    if warn_key == 'closest_pedestrian':
                        if sum(closest_pedestrians) > len(closest_pedestrians)/2:
                            warn_status['closest_pedestrian'] = True
                        else:
                            warn_status['closest_pedestrian'] = False
                        if warn_status['closest_pedestrian']:
                            print_value = warn_key + ' : ' + 'Yes'
                            print_color = [255, 255, 255]
                        else:
                            print_value = warn_key + ' : ' + 'No'
                            print_color = [0, 0, 255]                                              

                    if warn_key == 'on_road':
                        if sum(on_roads) > len(on_roads)/2:
                            warn_status['on_road'] = True
                        else:
                            warn_status['on_road'] = False
                        if warn_status['on_road']:
                            print_value = warn_key + ' : ' + 'On'
                            print_color = [255, 255, 255]
                        else:
                            print_value = warn_key + ' : ' + 'Off'
                            print_color = [0, 0, 255]

                    if warn_key == 'speed':
                        print_value = warn_key + ' : ' + str(warn_status[warn_key])
                        print_color = [255, 255, 255]

                    if not opt.nodebug:
                        if warn_key not in ['ignore_light', 'crossline', 'on_sidewalk', 'ignore_pedestrain']:
                            tf = max(tl - 1, 1)  # font thickness
                            t_size = cv2.getTextSize(print_value, 0, fontScale=tl / 3, thickness=tf)[0]
                            c1 = (0, int(t_size[1]*2*warn_key_idx+3))
                            c2 = c1[0] + t_size[0], c1[1] + t_size[1]*2
                            cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                            cv2.putText(im0, print_value, (c1[0], c2[1] - int(t_size[1]/2)), 0, tl / 3, print_color, thickness=tf, lineType=cv2.LINE_AA)
                            warn_key_idx+=1
                
                if warn_status['last_light'] == 'Red' and warn_status['speed'] >= 0:
                    warn_status['ignore_light'] = [True, 30]
                if warn_status['ignore_light'][0]:
                    warn_status['ignore_light'][1] -= 1
                    if warn_status['ignore_light'][1] <= 0: 
                        warn_status['ignore_light'][0] = False
                        warn_status['ignore_light'][1] = 0

                if warn_status['crossline'][0]:
                    warn_status['crossline'][1] -= 1
                    if warn_status['crossline'][1] <= 0: 
                        warn_status['crossline'][0] = False
                        warn_status['crossline'][1] = 0

                if not warn_status['on_road']:
                    warn_status['on_sidewalk'] = [True, 30]
                if warn_status['on_sidewalk'][0]:
                    warn_status['on_sidewalk'][1] -= 1
                    if warn_status['on_sidewalk'][1] <= 0: 
                        warn_status['on_sidewalk'][0] = False
                        warn_status['on_sidewalk'][1] = 0

                if warn_status['closest_pedestrian']:
                    warn_status['ignore_pedestrain'] = [True, 30]
                if warn_status['ignore_pedestrain'][0]:
                    warn_status['ignore_pedestrain'][1] -= 1
                    if warn_status['ignore_pedestrain'][1] <= 0: 
                        warn_status['ignore_pedestrain'][0] = False
                        warn_status['ignore_pedestrain'][1] = 0
                        

                for warn_key in warn_status.keys():
                    if warn_key == 'ignore_light':
                        if warn_status[warn_key][0]:
                            print_value = warn_key + ' : ' + 'Yes'
                            print_color = [0, 0, 255]
                        else:
                            print_value = warn_key + ' : ' + 'No'
                            print_color = [255, 255, 255]
                        
                    if warn_key == 'crossline':
                        if warn_status[warn_key][0]:
                            print_value = warn_key + ' : ' + 'Yes'
                            print_color = [0, 0, 255]
                        else:
                            print_value = warn_key + ' : ' + 'No'
                            print_color = [255, 255, 255]

                    if warn_key == 'on_sidewalk':
                        if warn_status[warn_key][0]:
                            print_value = warn_key + ' : ' + 'Yes'
                            print_color = [0, 0, 255]
                        else:
                            print_value = warn_key + ' : ' + 'No'
                            print_color = [255, 255, 255]

                    if warn_key == 'ignore_pedestrain':
                        if warn_status[warn_key][0]:
                            print_value = warn_key + ' : ' + 'Yes'
                            print_color = [0, 0, 255]
                        else:
                            print_value = warn_key + ' : ' + 'No'
                            print_color = [255, 255, 255]
                    
                    if warn_key in ['ignore_light', 'crossline', 'on_sidewalk', 'ignore_pedestrain']:
                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(print_value, 0, fontScale=tl / 3, thickness=tf)[0]
                        c1 = (0, int(t_size[1]*2*warn_key_idx+3))
                        c2 = c1[0] + t_size[0], c1[1] + t_size[1]*2
                        cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                        cv2.putText(im0, print_value, (c1[0], c2[1] - int(t_size[1]/2)), 0, tl / 3, print_color, thickness=tf, lineType=cv2.LINE_AA)
                        warn_key_idx+=1


                # Print time (inference + NMS)
                # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

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
                                cv2.imwrite(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0])+'_'+'0'*(6-len(str(frame)))+str(frame)+'.jpg', im0)
                                cv2.imwrite(os.path.join(save_dir, 'images', p.name.split('.')[0])+'_'+'0'*(6-len(str(frame)))+str(frame)+'_clean.jpg', clean_im0)
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
                            cv2.imwrite(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0])+'_'+'0'*(6-len(str(frame)))+str(frame)+'.jpg', im0)
                            if len(det) > 0:
                                cv2.imwrite(os.path.join(save_dir, 'images', p.name.split('.')[0])+'_'+'0'*(6-len(str(frame)))+str(frame)+'_clean.jpg', clean_im0)
                        vid_writer.write(im0)
        print(f'Each. ({time.time() - t0_each:.3f}s)')

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
    parser.add_argument('--calc-optical-flow', action='store_true', help='save each frame of video results')
    parser.add_argument('--nodebug', action='store_true', help='save each frame of video results')
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
