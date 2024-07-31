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
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import os
import numpy as np
from collections import defaultdict

import torchvision.transforms as transforms
import random

from torchvision.models import resnet101
from PIL import Image
import math

CAMERA_LENS_ANGLE = 57
CAMERA_RATIO = math.tan((CAMERA_LENS_ANGLE*math.pi/180)/2)*2 # 57 -> 1.0859113992768736 # 90 -> 2.0
OVERALL_HEIGHT_DICT = { # meter
    0 : 1.8, # 승용차, 차량류는 각도 등에 의해 박스 크기는 실제 전고보다 길 것을 반영하여 조금 더 크게 잡는다
    1 : 3.0, # 대형차, 종류가 다양해서 가장 차이 심함
    2 : 1.2, # 이륜차, 사람은 포함하지 않으니 더 줄어듦
    3 : 1.6, # 사람, 길지 않아서 왜곡 정도 낮음, 그러나 자세 등으로 오히려 작아질 가능성 큼
}

def _iou(A,B):
    low = np.s_[...,:2]
    high = np.s_[...,2:4]
    A,B = A[:, None].copy(),B[None].copy()
    A[high] += 1; B[high] += 1
    intrs = (np.maximum(0,np.minimum(A[high],B[high])
                        -np.maximum(A[low],B[low]))).prod(-1)
    return intrs / ((A[high]-A[low]).prod(-1)+(B[high]-B[low]).prod(-1)-intrs)

def calculate_distance(cxywh):
    c = cxywh[0]
    x = cxywh[1]
    y = cxywh[2]
    w = cxywh[3]
    h = cxywh[4]

    pixel_ratio = h

    distance = (OVERALL_HEIGHT_DICT[c]) / (pixel_ratio) / CAMERA_RATIO # 거리 (m)
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
    source, weights, weights_person, view_img, save_txt, imgsz = opt.source, opt.weights, opt.weights_person, opt.view_img, opt.save_txt, opt.img_size
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

    # Load model
    model_person = attempt_load(weights_person, map_location=device)  # load FP32 model
    if len(opt.img_size)==2:
        model_person = TracedModel(model_person, device, tuple(opt.img_size))
    elif len(opt.img_size)==1:
        model_person = TracedModel(model_person, device, tuple([opt.img_size[0], opt.img_size[0]]))
        

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
    names = ['car-mid', 'car-large', 'bike', 'person']#model.module.names if hasattr(model, 'module') else model.names
    colors = [[20,20,255], [20,255,20], [255,20,20], [255,20,255], [20,255,255], [255,255,20]]
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once

    if opt.save_frame:
        os.makedirs(os.path.join(save_dir, 'vis_frames'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)

    prev_boxes_list = [np.array([]) for _ in range(5)]
    prev_distances_list = [[] for _ in range(5)]

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
        pred = model(img, augment=opt.augment)[0]
        pred_person = model_person(img, augment=opt.augment)[0]

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred_person = non_max_suppression(pred_person, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred_person[0][:, 5] = 3

        if len(pred[0]) and len(pred_person[0]):
            pred = [torch.cat((pred[0], pred_person[0]), 0)]
        elif len(pred_person[0]):
            pred = pred_person

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
            max_person_size = 0 # (1080 x 1920)
            closest_pedestrian_distance = -1
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    scores = det[:, 4]
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)

                    temp_boxes = det[:, :4].detach().cpu().float().numpy()
                    if len(prev_boxes) and len(temp_boxes):
                        iou_matrix = _iou(prev_boxes, temp_boxes)
                    else:
                        iou_matrix = np.array([])
                    
                    if len(iou_matrix):
                        thr_score = 0.5
                        valid_idx = np.nonzero(np.max(iou_matrix, axis=0)>=thr_score)[0]
                        match_box_ids_y = np.argmax(iou_matrix, axis=0)
                        print("temp_boxes: \n", temp_boxes)
                        print("prev_boxes: \n", prev_boxes)
                        print("iou_matrix: \n", iou_matrix)
                        print("match_box_ids_y: \n", match_box_ids_y)
                        print("valid_idx: \n", valid_idx)
                    else:
                        valid_idx = np.array([])
                        match_box_ids_y = np.array([])

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    # Write results
                    temp_distances = []
                    print("prev_boxes: \n", prev_boxes)
                    print("prev_distances_list: \n", prev_distances_list)
                    for temp_box_id, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                        print(temp_box_id)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        if save_txt:  # Write to file                                
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        temp_distance = calculate_distance([int(cls), float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])])
                        temp_distances.append(temp_distance)
                        
                        if temp_box_id in valid_idx:
                            prev_box_id = match_box_ids_y[temp_box_id]
                            prev_distance = prev_distances[prev_box_id]
                            distance = round(prev_distance*0.34 + temp_distance*0.66, 3)

                            xyxy = np.array([prev_boxes[prev_box_id][0]*0.34 + xyxy[0].detach().cpu().float()*0.66,
                                             prev_boxes[prev_box_id][1]*0.34 + xyxy[1].detach().cpu().float()*0.66,
                                             prev_boxes[prev_box_id][2]*0.34 + xyxy[2].detach().cpu().float()*0.66,
                                             prev_boxes[prev_box_id][3]*0.34 + xyxy[3].detach().cpu().float()*0.66])
                        else:
                            distance = round(temp_distance, 3)

                        if save_img or view_img:  # Add bbox to image
                            distance_str = str(distance) + 'm'
                            if opt.debug:
                                label = f'{names[int(cls)]} {conf:.2f} {distance_str}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                            else:
                                label = distance_str
                                plot_one_box(xyxy, im0, label=label, color=[64, 128, 16], line_thickness=3)

                    
                    prev_boxes = temp_boxes
                    prev_distances = temp_distances
                else:
                    prev_boxes = np.array([])
                    prev_distances = []


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--weights-person', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
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
    parser.add_argument('--debug', action='store_true', help='turn on debug mode')
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
