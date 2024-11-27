import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load, attempt_load_v2
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, process_mask, scale_masks, process_semantic_mask
from utils.plots import plot_one_box, plot_masks
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from models.yolo import Model
import collections
import json
import os
import numpy as np
from collections import defaultdict

from sort import Sort

def _iou(A,B):
    low = np.s_[...,:2]
    high = np.s_[...,2:4]
    A,B = A[:, None].copy(),B[None].copy()
    A[high] += 1; B[high] += 1
    intrs = (np.maximum(0,np.minimum(A[high],B[high])
                        -np.maximum(A[low],B[low]))).prod(-1)
    return intrs / ((A[high]-A[low]).prod(-1)+(B[high]-B[low]).prod(-1)-intrs)

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0), colors=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        color = colors[int(cat)] if identities is not None else (255,0,20)
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
    return img

def update_tracker(sort_tracker, det, min_age):
    #im0 : type: numpy, shape: [width, height, 3]
    #det : type: torch.Tensor, shape: [box_num, 6] (6 for [x1, y1, x2, y2, confidence_score, class_id])   
    dets_to_sort = np.empty((0,6))
    
    for x1,y1,x2,y2,conf,detclass in det:
        dets_to_sort = np.vstack((dets_to_sort, 
                    np.array([x1, y1, x2, y2, conf, detclass])))
    
    # Run SORT
    tracked_dets, dead_trackers = sort_tracker.update(dets_to_sort)
    tracks = sort_tracker.getTrackers()

    #loop over tracks
    track_centroids_list = []
    for track in tracks:
        if track.age > min_age:
            track_centroids = track.centroidarr[min_age:]
            track_centroids_list.append(track_centroids)
    
    return tracked_dets, track_centroids_list

def visualize_tracker(im0, tracked_dets, track_centroids_list, names, colors):
    for track_centroids in track_centroids_list:        
        #draw colored tracks
        [cv2.line(im0, (int(track_centroids[i][0]),
                        int(track_centroids[i][1])), 
                        (int(track_centroids[i+1][0]),
                        int(track_centroids[i+1][1])),
                        (255,0,0), thickness=2) 
                        for i,_ in  enumerate(track_centroids) 
                        if i < len(track_centroids)-1 ]
            
    # draw boxes for visualization
    if len(tracked_dets)>0:
        bbox_xyxy = tracked_dets[:,:4]
        identities = tracked_dets[:, 5]
        categories = tracked_dets[:, 4]
        draw_boxes(im0, bbox_xyxy, identities, categories, names, colors=colors)
    
    return im0

def detect(save_img=False):
    #Initialize SORT
    sort_max_age = 9 # negative age means infinite 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    min_age = 3
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    tracked_bear_ids = []

    bbox_num = 0
    bbox_num_per_cls = defaultdict(int)
    '''
    bbox_num_per_size = {}
    for size_ratio_div in range(0,200,4):
        bbox_num_per_size[size_ratio_div/100000] = 0
    bbox_num_per_size[1] = 0
    print(bbox_num_per_size)
    '''
    source, weights, view_img, save_txt, imgsz, trace, qat = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.qat
    cfg, nc = opt.cfg, opt.nc
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
    print("HALF: ", half)

    # Load model
    if cfg and nc:
        nc = nc
        ckpt_weights = torch.load(weights[0], map_location=device)  # load checkpoint

        if type(ckpt_weights['ema']) is Model:
            state_dict = ckpt_weights['ema'].float().state_dict()  # to FP32
        elif type(ckpt_weights['ema']) is collections.OrderedDict:
            state_dict = ckpt_weights['ema']  # to FP32
        else:
            assert (type(ckpt_weights['ema']) is Model or type(ckpt_weights['ema']) is collections.OrderedDict), "Invalid model types to load"

        ckpt = Model(cfg, ch=3, nc=nc, qat=qat).to(device)  # create#, nm=nm).to(device)  # create
        #if opt.qat:
        #    fuse_modules(ckpt)
        model = attempt_load_v2(ckpt, state_dict, map_location=device)  # load FP32 model
    else:
        assert not cfg and not nc
        model = attempt_load(weights, map_location=device)  # load FP32 model
        if not hasattr(model, 'qat'):
            setattr(model, 'qat', False)
            
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    #imgsz = check_img_size(imgsz, s=stride)  # check img_size

    square_size = min(opt.img_size)

    if trace:
        if opt.square:
            model = TracedModel(model, device, tuple([square_size, square_size]))
        elif len(opt.img_size)==2:
            model = TracedModel(model, device, tuple(opt.img_size))
        elif len(opt.img_size)==1:
            model = TracedModel(model, device, tuple([opt.img_size[0], opt.img_size[0]]))

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, ratio_maintain=(not opt.no_ratio_maintain))

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    
    colors = [[255, 50, 50], [50, 255, 50], [50, 50, 255], [30,125,255], [255, 200, 100], 
    [100, 255, 200], [200, 100, 255], [200, 0, 0], [0, 200, 0], [0, 0, 200], 
    [0, 0, 0], [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 255, 255]]
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if opt.square:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, square_size, square_size).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = square_size
        old_img_h = square_size
    else:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = imgsz[1]
        old_img_h = imgsz[0]
    old_img_b = 1

    jdict = []

    if opt.save_frame:
        os.makedirs(os.path.join(save_dir, 'vis_frames'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'images_detected'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'images_nothing'), exist_ok=True)

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if img is None:
            continue
        print(img.shape)
        if opt.square:
            if img.shape[1] == square_size:
                square_crop_margin = int((img.shape[2] - square_size) / 2)
                img = img[:, :, square_crop_margin : square_crop_margin+square_size]
            elif img.shape[2] == square_size:
                square_crop_margin = int((img.shape[1] - square_size) / 2)
                img = img[:, square_crop_margin : square_crop_margin+square_size, :]
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
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred, out = model(img, augment=opt.augment)
        if opt.objcam:
            obj1 = (out[0][0, :, :, :, 4]).sigmoid().cpu().numpy()*255/3
            obj2 = (out[1][0, :, :, :, 4]).sigmoid().cpu().numpy()*255/3
            obj3 = (out[2][0, :, :, :, 4]).sigmoid().cpu().numpy()*255/3

        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
            
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)            

            short_side = min(im0.shape[0], im0.shape[1])
            if opt.square:
                if im0.shape[0] == short_side:
                    square_crop_margin = int((im0.shape[1] - short_side) / 2)
                    im0 = im0[:, square_crop_margin : square_crop_margin+short_side, :]
                elif im0.shape[1] == short_side:
                    square_crop_margin = int((im0.shape[0] - short_side) / 2)
                    im0 = im0[square_crop_margin : square_crop_margin+short_side, :, :]
            clean_im0 = im0.copy()

            if opt.objcam:
                alpha = 0.4
                view_cam = im0.copy()
                obj1 = cv2.resize(np.sum(obj1, 0).astype(np.uint8), (im0.shape[1], im0.shape[0]), interpolation = cv2.INTER_NEAREST)
                obj2 = cv2.resize(np.sum(obj2, 0).astype(np.uint8), (im0.shape[1], im0.shape[0]), interpolation = cv2.INTER_NEAREST)
                obj3 = cv2.resize(np.sum(obj3, 0).astype(np.uint8), (im0.shape[1], im0.shape[0]), interpolation = cv2.INTER_NEAREST)
                
                view_cam[:,:,2] = obj1
                view_cam[:,:,1] = obj2
                view_cam[:,:,0] = obj3
                im0 = cv2.addWeighted(im0, 0.9, view_cam, 0.6, 0)

            if opt.frame_ratio <= 0:
                frame_ratio = fps

            if dataset.mode != 'image' and frame % frame_ratio != 0:
                continue
            else:
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else '_'+'0'*(6-len(str(frame)))+str(frame))   # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                gn_to = torch.tensor(img.shape)[[3, 2, 3, 2]] 
                if len(det):
                    print(det)

                    scores = det[:, 4]
                    # Rescale boxes from img_size to im0 size
                    if opt.no_ratio_maintain:
                        det[:, 0] = det[:, 0] * (im0.shape[1] / img.shape[3])
                        det[:, 1] = det[:, 1] * (im0.shape[0] / img.shape[2])
                        det[:, 2] = det[:, 2] * (im0.shape[1] / img.shape[3])
                        det[:, 3] = det[:, 3] * (im0.shape[0] / img.shape[2])
                    else:
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)#.round()d

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string                        
                                            
                    # Write results
                    tracked_dets, track_centroids_list = update_tracker(sort_tracker, reversed(det[:, :6]).detach().cpu().numpy(), min_age=min_age) 
                    im0 = visualize_tracker(im0, tracked_dets, track_centroids_list, names, colors)
                    identities = tracked_dets[:, 5]
                    print("identities: ", identities)
                    for box_id, (*xyxy, cls, identity) in enumerate(tracked_dets[:, :6]):
                        if cls == 0:
                            if len(tracked_dets[tracked_dets[:, 4]==1, :4]) > 0:
                                tracking_iou_matrix = _iou(np.expand_dims(tracked_dets[box_id, :4], 0), 
                                        tracked_dets[tracked_dets[:, 4]==1, :4])
                                tracking_iou_thr = 0.05
                                print(tracking_iou_matrix)
                                if np.max(tracking_iou_matrix)>=tracking_iou_thr:
                                    tracked_bear_ids.append(identity)

                    for box_id, (*xyxy, cls, identity) in enumerate(reversed(tracked_dets[:, :6])):
                        if cls == 0 and identity in tracked_bear_ids:
                            box_marin = 15
                            plot_one_box([int(xyxy[0]-box_marin), int(xyxy[1]-box_marin), int(xyxy[2]+box_marin), int(xyxy[3]+box_marin)], im0, label="BEAR", color=[0, 0, 255], line_thickness=3)

                        if (xyxy[3]-xyxy[1]) < opt.min_size:
                            continue
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = cls, *xywh  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            size = (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])
                            label = f'{names[int(cls)]}'
                        
                        if opt.save_json:
                            if dataset.mode == 'image':
                                if int(cls.detach().cpu().numpy())==0:
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                                    line = (cls, xywh)  # label format
                                    image_id = p.name.split(".")[0]
                                    try: 
                                        int(image_id)
                                        image_id = int(image_id)
                                    except ValueError:
                                        image_id = image_id
                                    jdict_item = {
                                                'bbox': [x for x in line[1]],
                                                'category_id': 1,
                                                'image_id': image_id}
                                    jdict.append(jdict_item)
                            else:
                                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                                #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                                xyxy_ = [float((item / gn[idx] * gn_to[idx]).detach().cpu().numpy()) for idx, item in enumerate(xyxy)]
                                line = (cls, xyxy_)  # label format
                                jdict_item = {'image_id': frame,
                                            'category_id': int(line[0].detach().cpu().numpy())+1,
                                            'bbox': [x for x in line[1]],
                                            'video_path': path.split("/")[-1]}
                                jdict.append(jdict_item)
                        bbox_num+=1
                        bbox_num_per_cls[names[int(cls)]]+=1
                else:
                    tracked_bear_ids = []

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
                                cv2.imwrite(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0])+'.jpg', im0)
                                cv2.imwrite(os.path.join(save_dir, 'images_detected', p.name.split('.')[0])+'.jpg', clean_im0)
                            else:
                                cv2.imwrite(os.path.join(save_dir, 'images_nothing', p.name.split('.')[0])+'.jpg', clean_im0)
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
                            if opt.square:
                                w = min(w, h)
                                h = min(w, h)
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        if opt.save_frame:
                            print(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0]))
                            if len(det) > 0:
                                cv2.imwrite(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0])+'_'+'0'*(6-len(str(frame)))+str(frame)+'.jpg', im0)
                                cv2.imwrite(os.path.join(save_dir, 'images_detected', p.name.split('.')[0])+'_'+'0'*(6-len(str(frame)))+str(frame)+'.jpg', clean_im0)
                            else:
                                cv2.imwrite(os.path.join(save_dir, 'images_nothing', p.name.split('.')[0])+'_'+'0'*(6-len(str(frame)))+str(frame)+'.jpg', clean_im0)
                               
                        vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
    
    # Save JSON
    jdict.reverse()
    if opt.save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nsaving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

    print(f'Done. ({time.time() - t0:.3f}s)')
    print("BBOX NUM: ", bbox_num)
    for k, v in bbox_num_per_cls.items():
        print(k, " : ", v)
    #for k, v in dict(sorted(bbox_num_per_size.items())).items():
    #    print(k, " (", int(k*(im0.shape[0]*im0.shape[1])), ") : ", v)
        
    #with open('bboxnum.txt', 'a') as f:
    #    f.write(str(bbox_num) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--save-npy', action='store_true', help='save npy files')
    parser.add_argument('--square', action='store_true', help='do square cut for input')
    parser.add_argument('--objcam', action='store_true', help='visualize extracted objectness scores.')
    parser.add_argument('--no-ratio-maintain', action='store_true', help='maintain input ratio')
    parser.add_argument('--min-size', default=0, type=int, help='save frame ratio')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--nc', type=int, default=0, help='number of class')
    parser.add_argument('--qat', action='store_true', help='Quantization-Aware-Training')
    
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
