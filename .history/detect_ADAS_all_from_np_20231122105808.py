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
    last_light_length = 180
    warn_status = {
        'last_light' : ['Nolight']*last_light_length,
        'closest_pedestrian' : -1,
        'on_road' : False,
        'crossline' : [False, 0],
        'speed' : 0
    }
    lane_x_medians = [None, None, None, None, None, None]
    prev_gray = None
    bbox_num = 0
    bbox_num_per_cls = defaultdict(int)
    
    results, view_img, save_txt, imgsz = opt.results, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

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

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    road_indices = [13]
    seg_colors = [[255, 100, 100]]#[[random.randint(0, 255) for _ in range(3)] for _ in range(len(road_indices))]
    lane_colors = [[255, 100, 255], [250, 250, 250], [100, 255, 100], [100, 255, 255], [100, 100, 255]]#[[random.randint(0, 255) for _ in range(3)] for _ in range(5)]

    old_img_w = imgsz[1]
    old_img_h = imgsz[0]
    old_img_b = 1

    img_source = os.path.join(results, 'images')
    lane_source = os.path.join(results, 'mask_npy', 'lane')
    road_source = os.path.join(results, 'mask_npy', 'drivable_area')
    det_source = os.path.join(results, 'labels')


    if opt.save_frame:
        os.makedirs(os.path.join(save_dir, 'vis_frames'), exist_ok=True)

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(img_source, img_size=imgsz, stride=stride)

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
        
        t2 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
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
            lane_np = np.load(os.path.join(lane_source, p.name.split('/')[-1].split('.')[0]+'.npy'))

            lane_x_median = None
            for cls_idx in range(5):
                cls_pred_dilated = lane_np[cls_idx]
                cls_pred_dilated = cv2.resize(cls_pred_dilated, (im0.shape[1], im0.shape[0]), interpolation = cv2.INTER_NEAREST)

                vis_mask = im0.copy()
                vis_mask[cls_pred_dilated!=0] = np.array(lane_colors[cls_idx])
                alpha = 0.5
                im0 = cv2.addWeighted(im0, alpha, vis_mask, 1 - alpha, 0)

                if cls_idx == 3: # yellow line
                    lane_roi = cls_pred_dilated[int(im0.shape[0]*2/3):, int(im0.shape[1]*2/5):int(im0.shape[1]*3/5)]
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


            warn_status['on_road'] = False
            road_np = np.load(os.path.join(road_source, p.name.split('/')[-1].split('.')[0]+'.npy'))
            image_masks = cv2.resize(road_np, (im0.shape[1], im0.shape[0]), interpolation = cv2.INTER_NEAREST)            
            
            vis_mask = im0.copy()
            for road_idx, image_mask_idx in enumerate(road_indices):
                vis_mask[image_masks==image_mask_idx] = np.array(seg_colors[road_idx])
            alpha = 0.5
            im0 = cv2.addWeighted(im0, alpha, vis_mask, 1 - alpha, 0)

            on_road_roi = image_masks[int(image_masks.shape[0]/2):, int(image_masks.shape[1]/3):int(image_masks.shape[1]*2/3)]
            on_road_ratio = np.sum(on_road_roi==13) / (on_road_roi.shape[0]*on_road_roi.shape[1])
            if on_road_ratio > 0.3:
                warn_status['on_road'] = True
    
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
                            for lane_med_idx in range(len(warn_status['last_light'])-1):
                                warn_status['last_light'][lane_med_idx] = warn_status['last_light'][lane_med_idx+1]
                            warn_status['last_light'][-1] = last_light
                            
                            if cls == 4: # person
                                if xyxy[3] - xyxy[1] > max_person_size:
                                    max_person_size = xyxy[3] - xyxy[1]
                                    closest_pedestrian_distance = calculate_distance([int(cls), float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])])
                                    warn_status['closest_pedestrian'] = closest_pedestrian_distance

                if warn_status['crossline'][0]:
                    warn_status['crossline'][1] -= 1
                    if warn_status['crossline'][1] <= 0: 
                        warn_status['crossline'][0] = False
                        warn_status['crossline'][1] = 0

                tl = 3
                warn_key_idx = 0
                for warn_key in warn_status.keys():
                    if warn_key == 'last_light':
                        count_red = warn_status[warn_key].count('Red')
                        count_arrow = warn_status[warn_key].count('Arrow')
                        count_green = warn_status[warn_key].count('Green')
                        if warn_status[warn_key].count('Nolight') == len(warn_status['last_light']):
                            print_value = warn_key + ' : ' + 'Nolight'
                            print_color = [255, 255, 255]
                        else:
                            if count_red > count_arrow and count_red > count_green:
                                print_value = warn_key + ' : ' + 'Red'
                                print_color = [0, 0, 255]
                            elif count_green > count_red and count_green > count_arrow:
                                print_value = warn_key + ' : ' + 'Arrow'
                                print_color = [255, 100, 255]
                            elif count_arrow > count_red and count_arrow > count_green:
                                print_value = warn_key + ' : ' + 'Green'
                                print_color = [127, 255, 0]
                            else:
                                print_value = warn_key + ' : ' + 'Nolight'
                                print_color = [255, 255, 255]
                    
                    if warn_key == 'closest_pedestrian':
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
                        
                    if warn_key == 'crossline':
                        if warn_status[warn_key][0]:
                            print_value = warn_key + ' : ' + 'Yes'
                            print_color = [0, 0, 255]
                        else:
                            print_value = warn_key + ' : ' + 'No'
                            print_color = [255, 255, 255]

                    if warn_key == 'speed':
                        print_value = warn_key + ' : ' + str(warn_status[warn_key])
                        print_color = [255, 255, 255]

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
                            cv2.imwrite(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0])+'_'+str(frame)+'.jpg', im0)
                            if len(det) > 0:
                                cv2.imwrite(os.path.join(save_dir, 'images', p.name.split('.')[0])+'_'+str(frame)+'_clean.jpg', clean_im0)
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
    parser.add_argument('--results', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
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
