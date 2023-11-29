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
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import os
import numpy as np
from collections import defaultdict

import torchvision.transforms as transforms
import random

from torchvision.models import resnet101
from PIL import Image
import math

class_name_dict = {
    2 : 'trafficlight',
    4 : 'person',
    6 : 'car',
    7 : 'truck',
    8 : 'bus',
    10 : 'motorcycle',
}

COLORS = [[255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 255], [100, 255, 200], [200, 100, 255], [255, 200, 100], 
    [200, 0, 0], [0, 200, 0], [0, 0, 200], [0, 0, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255]]

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

    road_indices = [13]
    seg_colors = [[255, 100, 100]]#[[random.randint(0, 255) for _ in range(3)] for _ in range(len(road_indices))]
    lane_colors = [[255, 100, 255], [250, 250, 250], [100, 255, 100], [100, 255, 255], [100, 100, 255]]#[[random.randint(0, 255) for _ in range(3)] for _ in range(5)]

    old_img_w = imgsz[1]
    old_img_h = imgsz[0]
    old_img_b = 1

    img_source = os.path.join(results, 'images')
    det_source = os.path.join(results, 'labels')

    # 신호등 classification 모델 불러오기
    model_cls = resnet101(pretrained=True)
    model_cls.fc = nn.Linear(2048, 5)
    model_cls.load_state_dict(torch.load('resnet101_five_0048.pt'))
    model_cls = model_cls.to('cuda')
    model_cls.eval()

    # cls 모델 전처리 정의
    transform_cls = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if opt.save_frame:
        os.makedirs(os.path.join(save_dir, 'vis_frames'), exist_ok=True)

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(img_source, img_size=imgsz, stride=64)

    t0 = time.time()
    last_light = 'Nolight'
    print_color = [255, 255, 255]
    for path, img, im0s, vid_cap in dataset:
        t0_each = time.time()

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

        if dataset.mode != 'image' and frame % frame_ratio != 0:
            continue
        else:            
            if os.path.exists(os.path.join(det_source, p.name.split('/')[-1].split('_clean')[0]+'.txt')):
                with open(os.path.join(det_source, p.name.split('/')[-1].split('_clean')[0]+'.txt'), 'r') as f:
                    xyxy_lines = f.readlines()
                    for xyxy_line_idx, xyxy_line in enumerate(xyxy_lines):
                        if len(xyxy_line.strip().split(' ')) == 5:
                            c, x, y, w, h = xyxy_line.strip().split(' ')
                            cls = int(c)
                            if cls in class_name_dict:
                                x1_vis = (max(min((float(x)-float(w)/2)*im0.shape[1], im0.shape[1]), 0))
                                y1_vis = (max(min((float(y)-float(h)/2)*im0.shape[0], im0.shape[0]), 0))
                                x2_vis = (max(min((float(x)+float(w)/2)*im0.shape[1], im0.shape[1]), 0))
                                y2_vis = (max(min((float(y)+float(h)/2)*im0.shape[0], im0.shape[0]), 0))

                                if cls == 2: # traffic light
                                    margin = 32
                                    crop_x1 = int(x1_vis-margin)
                                    crop_y1 = int(y1_vis-margin)
                                    crop_x2 = int(x2_vis+margin)
                                    crop_y2 = int(y2_vis+margin)
                                    if crop_x1 > 0 and crop_y1 > 0 and crop_x2 < clean_im0.shape[1] and crop_y2 < clean_im0.shape[0]:
                                        traffic_light_crop = Image.fromarray(clean_im0.copy()[crop_y1:crop_y2, crop_x1:crop_x2, ::-1])
                                        traffic_light_crop = torch.unsqueeze(transform_cls(traffic_light_crop), 0).to('cuda')
                                        cls_outputs = model_cls(traffic_light_crop)
                                        _, predicted = torch.max(cls_outputs.data, 1)
                                        if predicted[0] == 0:
                                            last_light = 'Arrowgreen'
                                            print_color = [255, 100, 100]
                                        elif predicted[0] == 1:
                                            last_light = 'Red'
                                            print_color = [0, 0, 255]
                                        elif predicted[0] == 2:
                                            last_light = 'Arrow'
                                            print_color = [255, 100, 255]
                                        elif predicted[0] == 3:
                                            last_light = 'Green'
                                            print_color = [127, 255, 0]
                                        elif predicted[0] == 4:
                                            last_light = 'Yellow'
                                            print_color = [0, 255, 255]

                                im0 = cv2.rectangle(im0, (int(x1_vis), int(y1_vis)), (int(x2_vis), int(y2_vis)), COLORS[cls], thickness=3)

                                distance = str(calculate_distance([int(c), float(x), float(y), float(w), float(h)]))

                                class_name = class_name_dict[int(c)]
                                print_txt = class_name + " d=" + distance + "m"
                                tl = 3
                                tf = max(tl - 1, 1)  # font thickness
                                t_size = cv2.getTextSize(print_txt, 0, fontScale=tl / 3, thickness=tf)[0]
                                c1 = (int(x1_vis), int(y1_vis))
                                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                                cv2.putText(im0, print_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

                                if cls == 2: # traffic light
                                    print_txt = last_light
                                    tl = 3
                                    tf = max(tl - 1, 1)  # font thickness
                                    t_size = cv2.getTextSize(print_txt, 0, fontScale=tl / 3, thickness=tf)[0]
                                    c1 = (int(x1_vis), int(y1_vis) - t_size[1] - 3)
                                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                    cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                                    cv2.putText(im0, print_txt, (c1[0], c1[1] - 2), 0, tl / 3, print_color, thickness=tf, lineType=cv2.LINE_AA)


                    # Stream results
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if vid_path != save_path.split('_clean')[0].split('_'+save_path.split('_clean')[0].split('_')[-1])[0]:  # new video
                            vid_path = save_path.split('_clean')[0].split('_'+save_path.split('_clean')[0].split('_')[-1])[0]
                            last_light = 'Nolight'
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

    print(f'Done. ({time.time() - t0:.3f}s)')


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
    parser.add_argument('--calc-optical-flow', action='store_true', help='save each frame of video results')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    detect()
