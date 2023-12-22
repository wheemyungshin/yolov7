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
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.torch_utils import select_device, time_synchronized, TracedModel
from collections import defaultdict
import cv2
from utils.plots import plot_one_box
from numpy import random

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6):
    # 모델 불러오기
    device = select_device(opt.device, batch_size=batch_size)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size    
    model = TracedModel(model, device, imgsz)
    model.eval()
    

    # Config 불러오기
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)


    # Dataloader 만들기
    check_dataset(data)  # check
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = 'val'  # path to train/val/test images    
    valid_idx = data.get('valid_idx', None)
    pad_ratio = 0.5    
    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=pad_ratio, rect=True,
                                    prefix=colorstr(f'{task}: '), valid_idx=valid_idx)[0]


    # 평가 metric 설정하기
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()    
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
    jdict, ap, ap_class = [], [], []
    stats = []


    labels_f = open('det_results_raw_data.txt', 'w') 


    # Data 읽어서 Prediction 진행
    for batch_i, (img, targets, paths, shapes, _) in enumerate(tqdm(dataloader)):
        #데이터 불러와서 전처리
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # 모델 돌리기
            out, train_out = model(img, augment=False)  # inference and training outputs
            # NMS 적용
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=[], multi_label=True)

        for si, pred in enumerate(out):
            # 예측 결과값 후처리
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # 예측 결과를 이미지로 저장
            os.makedirs('det_results', exist_ok=True)
            im0 = cv2.imread(str(path.resolve()))
            for *xyxy, conf, cls in reversed(predn[:, :6]):
                if conf > 0.3 and names[int(cls)] in ['traffic light', 'person', 'car', 'truck', 'bus', 'motorcycle']:
                    size = (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])
                    label = f'{names[int(cls)]} {conf:.2f} {size}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    raw_data = str(path.resolve()).split('/')[-1] + ' : cls (' + str(label) + '), box (' + str(xyxy) + ')' + '\n'
                    labels_f.write(raw_data)
            cv2.imwrite(os.path.join('det_results', str(path.resolve()).split('/')[-1]), im0)

            # 평가
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                # GT 전처리
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                    if pi.shape[0]:
                        # GT와 예측 값 간의 IoU 계산
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        detected_set = set()

                        #IoU에 근거하여 일정 부분 이상 겹치면 정답 여부 체크
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            # 평가된 결과 저장
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # mAP 계산하기
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = torch.zeros(1)

    labels_f.close()

    # 결과 출력
    print("mAP@50 : ", map50*100, "%")


# 실행 및 option 설정
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)

    test(opt.data,
        opt.weights,
        opt.batch_size,
        opt.img_size,
        opt.conf_thres,
        opt.iou_thres
        )