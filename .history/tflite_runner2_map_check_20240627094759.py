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
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import time_synchronized, TracedModel
from collections import defaultdict
import cv2

import tensorflow as tf

def test(data,
         batch_size=32,
         imgsz=640,
         conf=0.001,
         iou_thres=0.6,  # for NMS
         dataloader=None,
         merge_label=[]):         
    gs = 32  

    fd_model = tf.lite.Interpreter(opt.weights)    
    fd_model2= tf.lite.Interpreter(opt.weights2)
    fd_model.allocate_tensors()
    fd_model2.allocate_tensors()

    # Configure
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    task = 'test'  # path to train/val/test images
    
    valid_idx = data.get('valid_idx', None)
    
    dataloader = create_dataloader(data[task], tuple([imgsz, imgsz]), batch_size, gs, opt, rect=False,
                                    prefix=colorstr(f'{task}: '), valid_idx=valid_idx, load_seg=False, ratio_maintain=True)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {id_: str(id_) for id_ in range(9999)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, ap, ap_class = [], [], []
    size_stats = None
    stats = []

    if len(merge_label) > 0:
        nc = len(merge_label)
        names = [str(n_num) for n_num in range(len(merge_label))]
    
    miou = [[] for _ in range(nc)]
    for batch_i, (img, targets, paths, shapes, _) in enumerate(tqdm(dataloader, desc=s)):
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height])  # to pixels
        #crop_img = np.expand_dims(crop_img.astype(np.float32), axis=0)
        img = img.permute(0,2,3,1)
        fd_model.set_tensor(fd_model.get_input_details()[0]['index'], img)
        fd_model.invoke()

        fd_output_0_0_ = fd_model.get_tensor(fd_model.get_output_details()[0]['index'])
        fd_output_0_1_ = fd_model.get_tensor(fd_model.get_output_details()[1]['index'])
        fd_output_0_2_ = fd_model.get_tensor(fd_model.get_output_details()[2]['index'])

        print(fd_model2.get_input_details()[0]["shape"], fd_output_0_0.shape)
        print(fd_model2.get_input_details()[1]["shape"], fd_output_0_1.shape)
        print(fd_model2.get_input_details()[2]["shape"], fd_output_0_2.shape)
        if fd_model2.get_input_details()[0]["shape"][1] == fd_output_0_0.shape[1]
        
        fd_model2.set_tensor(fd_model2.get_input_details()[0]['index'], fd_output_0_2)
        fd_model2.set_tensor(fd_model2.get_input_details()[1]['index'], fd_output_0_0)
        fd_model2.set_tensor(fd_model2.get_input_details()[2]['index'], fd_output_0_1)
        fd_model2.invoke()

        fd_output_1 = fd_model2.get_tensor(fd_model2.get_output_details()[0]['index'])
        fd_output_1 = fd_output_1[fd_output_1[:, -1] > conf]
        fd_output_1 = fd_output_1.clip(0, imgsz)

        out = np.zeros((fd_output_1.shape[0], 6))
        out[:, 0] = fd_output_1[:, 1]
        out[:, 1] = fd_output_1[:, 2]
        out[:, 2] = fd_output_1[:, 3]
        out[:, 3] = fd_output_1[:, 4]
        out[:, 4] = fd_output_1[:, 6]
        out[:, 5] = fd_output_1[:, 5]
        
        si = 0
        pred = torch.from_numpy(out)
        
        labels = targets[targets[:, 0] == si, 1:]            
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        path = Path(paths[si])
        seen += 1
        
        # Predictions
        predn = pred.clone()
        scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred	

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])

            scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
            
            confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1)) #plots

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            
                            if len(detected) == nl:  # all targets already located in image
                                break
                            
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, v5_metric=False, save_dir='', names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    
    #write history
    print("ap50: ", ap50[0])
    quantized_results = open(opt.result_txt+'.txt', 'a')    
    write_line = opt.weights + ' : ' + str(ap50[0])
    quantized_results.write(write_line)
    quantized_results.write('\n')
    quantized_results.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', type=str, default='../onnx2tf/saved_model/modified_phone_mobilenet_n78_tuning_only3_e089_no_opt_128_128_integer_quant.tflite', help='initial weights path')
    parser.add_argument('--weights2', type=str, default='../onnx2tf/saved_model/NMS_mobilenet_s128_128_float32.tflite', help='initial weights path')
    parser.add_argument('--save', type=str, default='n78_tuning3_phone_s128_e089_tel_c05.mp4', help='initial weights path')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=128, help='image sizes')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--merge-label', type=int, nargs='+', action='append', default=[], help='list of merge label list chunk. --merge-label 0 1 --merge-label 2 3 4')
    
    parser.add_argument('--result-txt', type=str, default='quantized_results', help='save name is *.txt')

    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    test(opt.data,
            opt.batch_size,
            opt.img_size,
            opt.conf,
            opt.iou_thres,
            merge_label=opt.merge_label,
            )