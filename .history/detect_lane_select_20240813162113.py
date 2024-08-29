import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, non_max_suppression_seg, process_mask, scale_masks, process_semantic_mask
from utils.plots import plot_one_box, plot_masks
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import json
import os

import numpy as np

from collections import defaultdict

def detect(save_img=False):
    bbox_num = 0
    bbox_num_per_cls = defaultdict(int)
    '''
    bbox_num_per_size = {}
    for size_ratio_div in range(0,200,4):
        bbox_num_per_size[size_ratio_div/100000] = 0
    bbox_num_per_size[1] = 0
    print(bbox_num_per_size)
    '''
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
    print("HALF: ", half)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
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
    print(names)
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if opt.seg:
        if len(opt.valid_segment_labels) > 0:
            nm = len(opt.valid_segment_labels)+1
            seg_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(max(opt.valid_segment_labels)+1)]
            seg_colors[13] = [100, 10, 10]
            seg_colors[14] = [100, 255, 255]
            seg_colors[15] = [255, 255, 100]
            seg_colors[16] = [225, 225, 225]
            seg_colors[17] = [255, 100, 255]
        else:
            nm = len(names)+1
            seg_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(nm)]

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
        if opt.seg:
            pred, out = model(img, augment=opt.augment)
            proto = out[1]
        else:
            pred, out = model(img, augment=opt.augment)
            if opt.objcam:
                obj1 = (out[0][0, :, :, :, 4]).sigmoid().cpu().numpy()*255/3
                obj2 = (out[1][0, :, :, :, 4]).sigmoid().cpu().numpy()*255/3
                obj3 = (out[2][0, :, :, :, 4]).sigmoid().cpu().numpy()*255/3

        t2 = time_synchronized()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #pred = non_max_suppression_seg(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, nm=len(names))#, nm=32)
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

                    if opt.seg:
                        #masks = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)
                        masks = process_semantic_mask(proto[i], det[:, 6:], det[:, :6], img.shape[2:], upsample=True)

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
                                            
                    # Mask plotting ----------------------------------------------------------------------------------------
                    if opt.seg:
                        #mcolors = [colors[int(cls)] for cls in det[:, 5]]
                        #im_masks = plot_masks(img[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                        #im0 = scale_masks(img.shape[2:], im_masks, im0.shape)  # scale to original h, w
                    
                        #for semantic masks
                        image_masks = masks.detach().cpu().numpy().astype(float)#[label_indexing]
                        
                        resize_ratio = im0.shape[1] / img.shape[3]
                        image_masks = image_masks[int((image_masks.shape[0]-(im0.shape[0]/resize_ratio))*2/3):-int((image_masks.shape[0]-(im0.shape[0]/resize_ratio))/3)]
                        image_masks = cv2.resize(image_masks, (im0.shape[1], im0.shape[0]), interpolation = cv2.INTER_NEAREST)

                        if opt.save_npy:
                            os.makedirs(os.path.join(save_dir, 'mask_npy'), exist_ok=True)
                            np.save(os.path.join(save_dir, 'mask_npy', p.name.split('.')[0]+'_'+'0'*(6-len(str(frame)))+str(frame)), image_masks)

                        vis_mask = im0.copy()
                        
                        for image_mask_idx in opt.valid_segment_labels:
                            vis_mask[image_masks==image_mask_idx] = np.array(seg_colors[image_mask_idx])
                        alpha = 0.5
                        im0 = cv2.addWeighted(im0, alpha, vis_mask, 1 - alpha, 0)
                    # Mask plotting ----------------------------------------------------------------------------------------
                    
                    # Write results
                    best_ratio_left = 0
                    best_ratio_left_idx = None
                    best_ratio_right = 0
                    best_ratio_right_idx = None
                    for i, lane_box in enumerate(det[:, :4]):                        
                        x = (lane_box[0]+lane_box[2])/2
                        y = (lane_box[1]+lane_box[3])/2
                        w = lane_box[2]-lane_box[0]
                        h = lane_box[3]-lane_box[1]
                        if x >= im0.shape[1]/2:
                            best_ratio_right = h/w if h/w > best_ratio_right else best_ratio_right
                            best_ratio_right_idx = i
                        else:
                            best_ratio_left = h/w if h/w > best_ratio_left else best_ratio_left
                            best_ratio_left_idx = i

                    det_i = len(det)-1
                    for (*xyxy, conf, cls) in reversed(det[:, :6]):
                        if len(opt.valid_segment_labels) > 0:
                            if cls-1 in opt.valid_segment_labels:
                                continue
                        if (xyxy[3]-xyxy[1]) < opt.min_size:
                            continue
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')


                        xyxy[0] = xyxy[0] - 3
                        xyxy[1] = xyxy[1] - 3
                        xyxy[2] = xyxy[2] + 3
                        xyxy[3] = xyxy[3] + 3

                        if save_img or view_img:  # Add bbox to image
                            size = (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])
                            label = f'{names[int(cls)]} {conf:.2f} {size}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        if best_ratio_left_idx is not None and det_i == best_ratio_left_idx:
                            cv2.line(im0, (int(xyxy[2]),int(xyxy[1])), (int(xyxy[0]),int(xyxy[3])), [255, 0, 255], 3)
                        if best_ratio_left_idx is not None and det_i == best_ratio_left_idx:
                            cv2.line(im0, (int(xyxy[2]),int(xyxy[1])), (int(xyxy[0]),int(xyxy[3])), [255, 0, 255], 3)
                                                
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
                                                'image_id': image_id,
                                                'score': float(conf.detach().cpu().numpy())}
                                    jdict.append(jdict_item)
                            else:
                                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                                #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                                xyxy_ = [float((item / gn[idx] * gn_to[idx]).detach().cpu().numpy()) for idx, item in enumerate(xyxy)]
                                line = (cls, xyxy_)  # label format
                                jdict_item = {'image_id': frame,
                                            'category_id': int(line[0].detach().cpu().numpy())+1,
                                            'bbox': [x for x in line[1]],
                                            'score': float(conf.detach().cpu().numpy()),
                                            'video_path': path.split("/")[-1]}
                                jdict.append(jdict_item)
                        bbox_num+=1
                        bbox_num_per_cls[names[int(cls)]]+=1

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
    parser.add_argument('--seg', action='store_true', help='Segmentation-Training')
    parser.add_argument('--save-npy', action='store_true', help='save npy files')
    parser.add_argument('--valid-segment-labels', nargs='+', type=int, default=[], help='labels to include when calculating segmentation loss')
    parser.add_argument('--square', action='store_true', help='do square cut for input')
    parser.add_argument('--objcam', action='store_true', help='visualize extracted objectness scores.')
    parser.add_argument('--no-ratio-maintain', action='store_true', help='maintain input ratio')
    parser.add_argument('--min-size', default=0, type=int, help='save frame ratio')
    
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
