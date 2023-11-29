import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from numpy import random

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, non_max_suppression_seg, process_mask, scale_masks, process_semantic_mask
from utils.plots import plot_one_box, plot_masks

import os
import numpy as np
from collections import defaultdict

import torchvision.transforms as transforms
import random

from PIL import Image
import math

#COLORS = [[255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 255], [100, 255, 200], [200, 100, 255], [255, 200, 100], 
#    [200, 0, 0], [0, 200, 0], [0, 0, 200], [0, 0, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
COLORS = [[255, 255, 255]]*20

class_name_dict = {
    2 : 'trafficlight',
    4 : 'person',
    6 : 'car',
    7 : 'truck',
    8 : 'bus',
    10 : 'motorcycle',
}

def _iou(A,B):
    low = np.s_[...,:2]
    high = np.s_[...,2:4]
    A,B = A[:, None].copy(),B[None].copy()
    intrs = (np.maximum(0,np.minimum(A[high],B[high])
                        -np.maximum(A[low],B[low]))).prod(-1)
    return intrs / ((A[high]-A[low]).prod(-1)+(B[high]-B[low]).prod(-1)-intrs)

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
    results, view_img, save_txt, imgsz = opt.results, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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

    road_indices = [13, 14, 15]
    seg_colors = [[255, 100, 100], [100, 255, 255], [255, 100, 255]]#[[random.randint(0, 255) for _ in range(3)] for _ in range(len(road_indices))]

    old_img_w = imgsz[1]
    old_img_h = imgsz[0]
    old_img_b = 1

    img_source = os.path.join(results, 'images')
    road_source = os.path.join(results, 'mask_npy')
    det_source = os.path.join(results, 'labels')

    if opt.save_frame:
        os.makedirs(os.path.join(save_dir, 'vis_frames'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'mask_npy'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)


    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(img_source, img_size=imgsz, stride=64)

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t0_each = time.time()
        img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        if opt.frame_ratio > 0:
            frame_ratio = opt.frame_ratio
        else:
            fps = 30

        # Process detections
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        clean_im0 = im0.copy()

        if opt.frame_ratio <= 0:
            frame_ratio = fps

            
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.name.split('_clean')[0]) + ('' if dataset.mode == 'image' else '_'+'0'*(6-len(str(frame)))+str(frame))  # img.txt
        frame_by_name = int(save_path.split('_clean')[0].split('_')[-1])
        if frame_by_name < 30:
            continue

        if frame_by_name > 250:
            continue

        if vid_path != save_path.split('_clean')[0].split('_'+save_path.split('_clean')[0].split('_')[-1])[0]:  # new video
            last_lights = ['Nolight']*120
            on_roads = [True]*150
            closest_pedestrians = [False]*30
            warn_status = {
                #for voting
                'last_light' : 'Nolight',
                'on_road' : False,
                'closest_pedestrian' : -1,
                'on_left' : [None, None],
                'going_left' : [None, None],
                'speed' : 0,

                #final
                'ignore_light' : [False, 0],
                'crossline' : [False, 0],
                'on_sidewalk' : [False, 0],
                'ignore_pedestrain' : [False, 0],
                'uturn' : [False, 0],
            }
            #lane_x_medians = [None]*6
            lane_on_lefts = [-1]*150
            lane_going_lefts = [-1]*150
            prev_gray = None
            prev_result = []

        if dataset.mode != 'image' and frame % frame_ratio != 0:
            continue
        else:
            on_road = False
            closest_pedestrian = False
            
            if os.path.exists(os.path.join(road_source, p.name.split('/')[-1].split('_clean')[0]+'.npy')):
                image_masks = np.load(os.path.join(road_source, p.name.split('/')[-1].split('_clean')[0]+'.npy'))          
                
                vis_mask = im0.copy()
                for road_idx, image_mask_idx in enumerate(road_indices):
                    vis_mask[image_masks==image_mask_idx] = np.array(seg_colors[road_idx])
                alpha = 0.5
                im0 = cv2.addWeighted(im0, alpha, vis_mask, 1 - alpha, 0)
            
                on_road_roi = image_masks[int(image_masks.shape[0]/2):, int(image_masks.shape[1]/3):int(image_masks.shape[1]*2/3)]
                on_road_ratio = np.sum(on_road_roi!=0) / (on_road_roi.shape[0]*on_road_roi.shape[1])
                if on_road_ratio > 0.3:
                    on_road = True

                np.save(os.path.join(save_dir, 'mask_npy', p.name.split('/')[-1].split('_clean')[0]+'.npy'), image_masks)

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
                    on_left = 1
                elif a > 0:
                    on_left = 0
                else:
                    on_left = -1
                if on_left is not None:
                    x_intercept = -b/a
                    if x_intercept > largest_mask.shape[1]/2:
                        going_left = 1
                    else:
                        going_left = 0
            else:
                on_left = -1
                going_left = -1

            for ongoing_left_idx in range(len(lane_on_lefts)-1):
                lane_on_lefts[ongoing_left_idx] = lane_on_lefts[ongoing_left_idx+1]
                lane_going_lefts[ongoing_left_idx] = lane_going_lefts[ongoing_left_idx+1]
            lane_on_lefts[-1] = on_left
            lane_going_lefts[-1] = going_left

            prev_lane_on_lefts = np.array(lane_on_lefts[:int(len(lane_on_lefts)/2)])
            temp_lane_on_lefts = np.array(lane_on_lefts[int(len(lane_on_lefts)/2):])
            lrth = 0.48
            if np.sum(prev_lane_on_lefts[prev_lane_on_lefts!=-1]) > len(prev_lane_on_lefts[prev_lane_on_lefts!=-1])*(1-lrth):
                warn_status['on_left'][0] = None
            elif np.sum(prev_lane_on_lefts[prev_lane_on_lefts!=-1]) < len(prev_lane_on_lefts[prev_lane_on_lefts!=-1])*lrth:
                warn_status['on_left'][0] = False
            else:
                warn_status['on_left'][0] = None
            if np.sum(temp_lane_on_lefts[temp_lane_on_lefts!=-1]) > len(temp_lane_on_lefts[temp_lane_on_lefts!=-1])*(1-lrth):
                warn_status['on_left'][1] = None
            elif np.sum(temp_lane_on_lefts[temp_lane_on_lefts!=-1]) < len(temp_lane_on_lefts[temp_lane_on_lefts!=-1])*lrth:
                warn_status['on_left'][1] = False
            else:
                warn_status['on_left'][1] = None

            prev_lane_going_lefts = np.array(lane_going_lefts[:int(len(lane_going_lefts)/2)])
            temp_lane_going_lefts = np.array(lane_going_lefts[int(len(lane_going_lefts)/2):])
            if np.sum(prev_lane_going_lefts[prev_lane_going_lefts!=-1]) > len(prev_lane_going_lefts[prev_lane_going_lefts!=-1])*(1-lrth):
                warn_status['going_left'][0] = None
            elif np.sum(prev_lane_going_lefts[prev_lane_going_lefts!=-1]) < len(prev_lane_going_lefts[prev_lane_going_lefts!=-1])*lrth:
                warn_status['going_left'][0] = False
            else:
                warn_status['going_left'][0] = None
            if np.sum(temp_lane_going_lefts[temp_lane_going_lefts!=-1]) > len(temp_lane_going_lefts[temp_lane_going_lefts!=-1])*(1-lrth):
                warn_status['going_left'][1] = None
            elif np.sum(temp_lane_going_lefts[temp_lane_going_lefts!=-1]) < len(temp_lane_going_lefts[temp_lane_going_lefts!=-1])*lrth:
                warn_status['going_left'][1] = False
            else:
                warn_status['going_left'][1] = None                    
            
            if os.path.exists(os.path.join(det_source, p.name.split('/')[-1].split('_clean')[0]+'.txt')):
                with open(os.path.join(det_source, p.name.split('/')[-1].split('_clean')[0]+'.txt'), 'r') as f:
                    xyxy_lines = f.readlines()
                    warn_status['speed'] = float(xyxy_lines[0].strip().split('Speed:')[-1])
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(xyxy_lines[0].strip() + '\n')
            
            last_light = 'Nolight'
            max_traffic_light_size = 0 
            max_person_size = 0
            closest_pedestrian_distance = -1
            
            if os.path.exists(os.path.join(det_source, p.name.split('/')[-1].split('_clean')[0]+'.txt')):
                temp_result = []
                with open(os.path.join(det_source, p.name.split('/')[-1].split('_clean')[0]+'.txt'), 'r') as f:
                    xyxy_lines = f.readlines()[1:]
                    for xyxy_line_idx, xyxy_line in enumerate(xyxy_lines):
                        if len(xyxy_line.strip().split(' ')) == 5:
                            c, x, y, w, h = xyxy_line.strip().split(' ')
                            cls = int(c)

                            x1_vis = (max(min((float(x)-float(w)/2)*im0.shape[1], im0.shape[1]), 0))
                            y1_vis = (max(min((float(y)-float(h)/2)*im0.shape[0], im0.shape[0]), 0))
                            x2_vis = (max(min((float(x)+float(w)/2)*im0.shape[1], im0.shape[1]), 0))
                            y2_vis = (max(min((float(y)+float(h)/2)*im0.shape[0], im0.shape[0]), 0))
                            if 160 <frame_by_name < 180 and float(y) > 0.5:
                                continue

                            if 180 <frame_by_name < 190 and float(y) > 0.7:
                                continue

                            if 130 <frame_by_name < 140 and float(x) < 0.15:
                                continue

                            if 110 <frame_by_name < 120 and float(x) < 0.1 and float(y) > 0.4:
                                continue

                            if (float(h)*im0.shape[0])*(float(w)*im0.shape[1]) > 1600:                                
                                #tracking
                                '''
                                prev_result_np = np.array(prev_result)
                                if len(prev_result_np) > 0:
                                    prev_result_matching_cls = prev_result_np[prev_result_np[:, 0]==cls][:, 1:]
                                    iou_matrix = _iou(np.array([[int(x1_vis), int(y1_vis), int(x2_vis), int(y2_vis)]]), prev_result_matching_cls)
                                    print(np.array([[int(x1_vis), int(y1_vis), int(x2_vis), int(y2_vis)]]))
                                    print(prev_result_matching_cls)
                                    print(iou_matrix)
                                    if len(iou_matrix[0]) > 0 and np.max(iou_matrix)> 0.3:
                                        im0 = cv2.rectangle(im0, (int(x1_vis), int(y1_vis)), (int(x2_vis), int(y2_vis)), COLORS[cls], thickness=3)
                                        temp_result.append([cls, int(x1_vis), int(y1_vis), int(x2_vis), int(y2_vis)])
                                else:
                                    im0 = cv2.rectangle(im0, (int(x1_vis), int(y1_vis)), (int(x2_vis), int(y2_vis)), COLORS[cls], thickness=3)
                                    temp_result.append([cls, int(x1_vis), int(y1_vis), int(x2_vis), int(y2_vis)])
                                '''
                                
                                if cls == 2:
                                    if float(w) > max_traffic_light_size:
                                        max_traffic_light_size = float(w)
                                        if xyxy_line_idx+1 < len(xyxy_lines):
                                            last_light = xyxy_lines[xyxy_line_idx+1].strip()
                                            
                                if int(c) in class_name_dict:
                                    im0 = cv2.rectangle(im0, (int(x1_vis), int(y1_vis)), (int(x2_vis), int(y2_vis)), COLORS[cls], thickness=3)
                                    distance = str(calculate_distance([int(c), float(x), float(y), float(w), float(h)]))
                                    class_name = class_name_dict[int(c)]
                                    print_txt ="d=" + distance + "m"
                                    tl = 2
                                    tf = max(tl - 1, 1)  # font thickness
                                    t_size = cv2.getTextSize(print_txt, 0, fontScale=tl / 3, thickness=tf)[0]
                                    c1 = (int(x1_vis), int(y1_vis))
                                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                    cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                                    cv2.putText(im0, print_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

                                    if cls == 2:
                                        line = [c, x, y, w, h, last_light]
                                    else:
                                        line = [c, x, y, w, h, 'Nolight']
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write((' '.join(line)).rstrip() + '\n')


                                if cls == 4: # person
                                    if float(h)*im0.shape[0] > max_person_size:
                                        max_person_size = float(h)*im0.shape[0]
                                        closest_pedestrian_distance = calculate_distance([cls, 
                                            float(x), float(y), float(w), float(h)])

                                        if closest_pedestrian_distance == -1:
                                            closest_pedestrian = False
                                        elif closest_pedestrian_distance > 2:
                                            closest_pedestrian = False
                                        else:
                                            closest_pedestrian = True
                prev_result = temp_result
                        
                for lane_med_idx in range(len(last_lights)-1):
                    last_lights[lane_med_idx] = last_lights[lane_med_idx+1]
                last_lights[-1] = last_light

                for on_road_idx in range(len(on_roads)-1):
                    on_roads[on_road_idx] = on_roads[on_road_idx+1]
                on_roads[-1] = on_road

                for ped_idx in range(len(closest_pedestrians)-1):
                    closest_pedestrians[ped_idx] = closest_pedestrians[ped_idx+1]
                closest_pedestrians[-1] = closest_pedestrian

                print(closest_pedestrians)
                print(warn_status)

                tl = 3
                warn_key_idx = 0
                for warn_key in warn_status.keys():
                    if warn_key == 'last_light':
                        count_red = last_lights.count('Red')
                        count_green = last_lights.count('Green')
                        count_yellow = last_lights.count('Yellow')
                        count_arrow = last_lights.count('Arrow')
                        count_arrowgreen = last_lights.count('Arrowgreen')
                        if last_lights.count('Nolight') == len(last_lights):
                            warn_status['last_light'] = 'Nolight'
                            print_color = [255, 255, 255]
                        else:
                            if  count_red > count_green and count_red > count_yellow and count_red > count_arrow and count_red > count_arrowgreen:
                                warn_status['last_light'] = 'Red'
                                print_color = [0, 0, 255]
                            elif count_green > count_red and count_green > count_yellow and count_green > count_arrow and count_green > count_arrowgreen:
                                warn_status['last_light'] = 'Green'
                                print_color = [127, 255, 0]
                            elif count_yellow > count_red and count_yellow > count_green and count_yellow > count_arrow and count_yellow > count_arrowgreen:
                                warn_status['last_light'] = 'Yellow'
                                print_color = [0, 255, 255]
                            elif count_arrow > count_red and count_arrow > count_green and count_arrow > count_yellow and count_arrow > count_arrowgreen:
                                warn_status['last_light'] = 'Arrow'
                                print_color = [255, 100, 255]
                            elif count_arrowgreen > count_red and count_arrowgreen > count_green and count_arrowgreen > count_yellow and count_arrowgreen > count_arrow:
                                warn_status['last_light'] = 'Arrowgreen'
                                print_color = [255, 100, 100]
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
                            print_color = [0, 0, 255]
                        else:
                            print_value = warn_key + ' : ' + 'No'
                            print_color = [255, 255, 255]                                    

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

                    if warn_key == 'on_left':
                        print_value = warn_key + ' : ' + str(warn_status[warn_key])
                        print_color = [255, 255, 255]

                    if warn_key == 'going_left':
                        print_value = warn_key + ' : ' + str(warn_status[warn_key])
                        print_color = [255, 255, 255]

                    if warn_key == 'speed':
                        print_value = warn_key + ' : ' + str(warn_status[warn_key])
                        print_color = [255, 255, 255]

                    if not opt.nodebug:
                        if warn_key not in ['ignore_light', 'crossline', 'on_sidewalk', 'ignore_pedestrain', 'uturn']:
                            tf = max(tl - 1, 1)  # font thickness
                            t_size = cv2.getTextSize(print_value, 0, fontScale=tl / 3, thickness=tf)[0]
                            c1 = (0, int(t_size[1]*2*warn_key_idx+3))
                            c2 = c1[0] + t_size[0], c1[1] + t_size[1]*2
                            cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                            cv2.putText(im0, print_value, (c1[0], c2[1] - int(t_size[1]/2)), 0, tl / 3, print_color, thickness=tf, lineType=cv2.LINE_AA)
                            warn_key_idx+=1
                
                if warn_status['last_light'] == 'Red' and last_lights[-30:].count('Nolight') == 30 and warn_status['speed'] >= 5:
                    warn_status['ignore_light'] = [True, 30]
                if warn_status['ignore_light'][0]:
                    warn_status['ignore_light'][1] -= 1
                    if warn_status['ignore_light'][1] <= 0: 
                        warn_status['ignore_light'][0] = False
                        warn_status['ignore_light'][1] = 0

                '''
                if warn_status['crossline'][0]:
                    warn_status['crossline'][1] -= 1
                    if warn_status['crossline'][1] <= 0: 
                        warn_status['crossline'][0] = False
                        warn_status['crossline'][1] = 0
                '''

                if not warn_status['on_road'] and warn_status['speed'] >= 5:
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

                if warn_status['on_left'][0] == False and warn_status['on_left'][1] == True :
                    warn_status['crossline'] = [True, 30]
                if warn_status['crossline'][0]:
                    warn_status['crossline'][1] -= 1
                    if warn_status['crossline'][1] <= 0: 
                        warn_status['crossline'][0] = False
                        warn_status['crossline'][1] = 0
                        
                if warn_status['on_left'][1] == False and warn_status['going_left'][1] == True :
                    warn_status['uturn'] = [True, 30]
                if warn_status['uturn'][0]:
                    warn_status['uturn'][1] -= 1
                    if warn_status['uturn'][1] <= 0: 
                        warn_status['uturn'][0] = False
                        warn_status['uturn'][1] = 0                        

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

                    if warn_key == 'uturn':
                        if warn_status[warn_key][0]:
                            print_value = warn_key + ' : ' + 'Yes'
                            print_color = [0, 0, 255]
                        else:
                            print_value = warn_key + ' : ' + 'No'
                            print_color = [255, 255, 255]
                    
                    if warn_key in ['ignore_light', 'crossline', 'on_sidewalk', 'ignore_pedestrain', 'uturn']:
                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(print_value, 0, fontScale=tl / 3, thickness=tf)[0]
                        c1 = (0, int(t_size[1]*2*warn_key_idx+3))
                        c2 = c1[0] + t_size[0], c1[1] + t_size[1]*2
                        cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                        cv2.putText(im0, print_value, (c1[0], c2[1] - int(t_size[1]/2)), 0, tl / 3, print_color, thickness=tf, lineType=cv2.LINE_AA)
                        warn_key_idx+=1

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if vid_path != save_path.split('_clean')[0].split('_'+save_path.split('_clean')[0].split('_')[-1])[0]:  # new video
                        vid_path = save_path.split('_clean')[0].split('_'+save_path.split('_clean')[0].split('_')[-1])[0]
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        vid_writer = cv2.VideoWriter(vid_path+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    if opt.save_frame:
                        cv2.imwrite(os.path.join(save_dir, 'vis_frames', p.name.split('_clean')[0])+'_'+str(frame)+'.jpg', im0)
                        cv2.imwrite(os.path.join(save_dir, 'images', p.name.split('_clean')[0])+'_'+str(frame)+'_clean.jpg', clean_im0)
                    vid_writer.write(im0)
        print(f'Each. ({time.time() - t0_each:.3f}s)')

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='runs/detect')
    parser.add_argument('--img-size', nargs='+', type=int, default=[192,256], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--frame-ratio', default=1, type=int, help='save frame ratio')
    parser.add_argument('--save-frame', action='store_true', help='save each frame of video results')
    parser.add_argument('--nodebug', action='store_true', help='save each frame of video results')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    detect()
