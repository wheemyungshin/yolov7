import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import json
import os


def detect(save_img=False):
    bbox_num = 0
    body_weights = opt.body_weights
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    body_model = attempt_load(body_weights, map_location=device)  # load FP32 model

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    #imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        body_model = TracedModel(body_model, device, tuple(opt.img_size))
        model = TracedModel(model, device, tuple(opt.img_size))

    if half:
        body_model.half()  # to FP16
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        body_model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(body_model.parameters())))  # run once
        model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = imgsz[1]
    old_img_h = imgsz[0]
    old_img_b = 1

    jdict = []


    if opt.save_frame:
        os.makedirs(os.path.join(save_dir, 'vis_frames'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'clean_frames'), exist_ok=True)


    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
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
                body_model(img, augment=opt.augment)[0]
                model(img, augment=opt.augment)[0]

        body_pred = body_model(img, augment=opt.augment)[0]
        body_pred = non_max_suppression(body_pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        largest_body = None
        largest_size = 0
        print(len(body_pred))
        for i, body_p in enumerate(body_pred[0]):
            size = (body_p[2]-body_p[0])*(body_p[3]-body_p[1])
            if size > largest_size:
                largest_size = size
                largest_body = torch.unsqueeze(body_p[:4], 0)
            
        if largest_body is not None:
            print(im0s.shape)
            print(largest_body)
            largest_body = scale_coords(img.shape[2:], largest_body, im0s.shape)[0]#.round()
            crop_img = im0s[int(largest_body[1]):int(largest_body[3]), int(largest_body[0]):int(largest_body[2]), :]
            cv2.imwrite('test.jpg', crop_img)
            print(crop_img.shape)
            resize_crop_img = cv2.resize(crop_img, (192, 192), interpolation=cv2.INTER_LINEAR)
            resize_crop_img_for_save = resize_crop_img.copy()
            resize_crop_img = np.ascontiguousarray(resize_crop_img[:, :, ::-1].transpose(2, 0, 1))
            resize_crop_img = torch.from_numpy(resize_crop_img).to(device)
            resize_crop_img = resize_crop_img.half() if half else resize_crop_img.float()  # uint8 to fp16/32
            resize_crop_img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if resize_crop_img.ndimension() == 3:
                resize_crop_img = resize_crop_img.unsqueeze(0)
            print(resize_crop_img.shape)

            # Inference
            t1 = time_synchronized()
            pred = model(resize_crop_img, augment=opt.augment)[0]
            t2 = time_synchronized()

            print(img.shape)
            print(pred)
            '''
            pred[:, :, 0] = ((pred[:, :, 0] * (crop_img.shape[1] / 192)) + int(largest_body[0])) / im0s.shape[1]
            pred[:, :, 1] = ((pred[:, :, 1] * (crop_img.shape[0] / 192)) + int(largest_body[1])) / im0s.shape[0]
            pred[:, :, 2] = ((pred[:, :, 2] * (crop_img.shape[1] / 192)) + int(largest_body[0])) / im0s.shape[1]
            pred[:, :, 3] = ((pred[:, :, 3] * (crop_img.shape[0] / 192)) + int(largest_body[1])) / im0s.shape[0]
            '''
        else:
            print("No body detected")
            exit()
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

        # Apply NMS
        preds = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        
        print(preds)
        print(path)
        print(resize_crop_img_for_save.shape)
        for pred in preds:
            if len(pred) > 0:
                pred = [min(pred[0][0], 188), min(pred[0][1], 188), min(pred[0][2], 188), min(pred[0][3], 188)]
                resize_crop_img_for_save[int(pred[1]):int(pred[3]), int(pred[2]):int(pred[2])+3, 2] = 255
                resize_crop_img_for_save[int(pred[1]):int(pred[3]), int(pred[0]):int(pred[0])+3, 2] = 255
                resize_crop_img_for_save[int(pred[1]):int(pred[1])+3, int(pred[0]):int(pred[2]), 2] = 255
                resize_crop_img_for_save[int(pred[3]):int(pred[3])+3, int(pred[0]):int(pred[2]), 2] = 255
        cv2.imwrite('test_imgs/'+path.split('/')[-1], resize_crop_img_for_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--body-weights', nargs='+', type=str, default='weights/0127memryx_mafa-yolov7-tiny_body_only_s192_320_B.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
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
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
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
