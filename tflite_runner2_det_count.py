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
from utils.datasets import LoadImages
import cv2

import tensorflow as tf

def output_matching(model2_input, fd_outputs):    
    if model2_input["shape"][1] == fd_outputs[0].shape[1]:
        fd_output = fd_outputs[0]
    elif model2_input["shape"][1] == fd_outputs[1].shape[1]:
        fd_output = fd_outputs[1]
    elif model2_input["shape"][1] == fd_outputs[2].shape[1]:
        fd_output = fd_outputs[2]
    else:
        print("wrong input shape: ", model2_input.shape)
    
    return fd_output

def test(data,
         batch_size=32,
         imgsz=640,
         conf=0.001,
         iou_thres=0.6,  # for NMS
         dataloader=None,
         merge_label=[]):         
    gs = 32  

    if len(imgsz) == 2:
        imgsz = [check_img_size(x, gs) for x in imgsz]  # verify imgsz are gs-multiples
        imgsz = tuple(imgsz)
    else:
        imgsz = check_img_size(imgsz[0], gs)  # verify imgsz are gs-multiples
        imgsz = tuple([imgsz, imgsz])
    
    print(imgsz)

    fd_model = tf.lite.Interpreter(opt.weights)    
    fd_model2= tf.lite.Interpreter(opt.weights2)
    fd_model.allocate_tensors()
    fd_model2.allocate_tensors()

    nc = 1

    # Dataloader
    task = 'test'  # path to train/val/test images
    
    dataset =  LoadImages(data, img_size=imgsz, stride=32, ratio_maintain=True)

    if len(merge_label) > 0:
        nc = len(merge_label)
        names = [str(n_num) for n_num in range(len(merge_label))]
    
    miou = [[] for _ in range(nc)]
    detected = 0
    for path, img, im0s, vid_cap in dataset:
        img = np.transpose(img, (1,2,0))
        img = (img / 255.0).astype(np.float32)  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img.astype(np.float32), axis=0)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run NMS
        #crop_img = np.expand_dims(crop_img.astype(np.float32), axis=0)
        fd_model.set_tensor(fd_model.get_input_details()[0]['index'], img)
        fd_model.invoke()

        fd_output_0_0_ = fd_model.get_tensor(fd_model.get_output_details()[0]['index'])
        fd_output_0_1_ = fd_model.get_tensor(fd_model.get_output_details()[1]['index'])
        fd_output_0_2_ = fd_model.get_tensor(fd_model.get_output_details()[2]['index'])

        fd_output_0_0 = output_matching(fd_model2.get_input_details()[0], [fd_output_0_0_, fd_output_0_1_, fd_output_0_2_])
        fd_output_0_1 = output_matching(fd_model2.get_input_details()[1], [fd_output_0_0_, fd_output_0_1_, fd_output_0_2_])
        fd_output_0_2 = output_matching(fd_model2.get_input_details()[2], [fd_output_0_0_, fd_output_0_1_, fd_output_0_2_])

        fd_model2.set_tensor(fd_model2.get_input_details()[0]['index'], fd_output_0_0)
        fd_model2.set_tensor(fd_model2.get_input_details()[1]['index'], fd_output_0_1)
        fd_model2.set_tensor(fd_model2.get_input_details()[2]['index'], fd_output_0_2)
        fd_model2.invoke()

        fd_output_1 = fd_model2.get_tensor(fd_model2.get_output_details()[0]['index'])
        fd_output_1 = fd_output_1[fd_output_1[:, -1] > conf]
        fd_output_1[:, 1] = fd_output_1[:, 1].clip(0, imgsz[1])
        fd_output_1[:, 2] = fd_output_1[:, 2].clip(0, imgsz[0])
        fd_output_1[:, 3] = fd_output_1[:, 3].clip(0, imgsz[1])
        fd_output_1[:, 4] = fd_output_1[:, 4].clip(0, imgsz[0])

        out = np.zeros((fd_output_1.shape[0], 6))
        out[:, 0] = fd_output_1[:, 1]
        out[:, 1] = fd_output_1[:, 2]
        out[:, 2] = fd_output_1[:, 3]
        out[:, 3] = fd_output_1[:, 4]
        out[:, 4] = fd_output_1[:, 6]
        out[:, 5] = fd_output_1[:, 5]
        
        si = 0
        pred = torch.from_numpy(out)
                
        if len(pred):
            detected += 1
        
    #write history
    print("ap50: ", detected)
    quantized_results = open(opt.result_txt+'.txt', 'a')    
    write_line = opt.weights + ' : (' + str(detected) + ')'
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
    parser.add_argument('--img-size', nargs='+', type=int, default=[128, 128], help='[height, width] image sizes')
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