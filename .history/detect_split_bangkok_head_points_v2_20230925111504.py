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

def check_head_in_body(box1, box2, margin=0):
    couples = []
    if box2[0] - margin <= box1[0] <= box2[2] + margin and \
        box2[1] - margin - (box2[3]-box2[1])*0.4 <= box1[1] <= box2[3] + margin and \
        box2[0] - margin <= box1[2] <= box2[2] + margin and \
        box2[1] - margin <= box1[3] <= box2[3] + margin:
        return True
    else:
        return False

def detect(save_img=False):
    bbox_num = 0
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
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    #imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        if len(opt.img_size)==2:
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
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = imgsz[1]
    old_img_h = imgsz[0]
    old_img_b = 1

    jdict = []

    if opt.save_frame:
        os.makedirs(os.path.join(save_dir, 'vis_frames'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'clean_frames'), exist_ok=True)

    frame_ratio = opt.frame_ratio
    fps = 25
    data_idx = -1
    real_run_idx = -1
    patch_num = list(opt.patch_num)
    pred_prev = {}
    for quadrant_idx in range(patch_num[0]*patch_num[1]):
        pred_prev[quadrant_idx] = None

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        data_idx+=1
        if data_idx % frame_ratio != 0:
            continue
        else:
            real_run_idx+=1
        if opt.all_quadrant_at_once:
            quadrant = -1
        else:
            quadrant = real_run_idx % (patch_num[0]*patch_num[1])
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
    
        pred_list = []
        margin = 64
        patch_stride_x = int((img.shape[3]-margin) / patch_num[1])
        patch_stride_y = int((img.shape[2]-margin) / patch_num[0])

        if quadrant == -1:
            pred_concat = None
            for y_i in range(patch_num[0]):
                for x_i in range(patch_num[1]):
                    x1 = check_img_size(x_i * patch_stride_x, 64)
                    y1 = check_img_size(y_i * patch_stride_y, 64)
                    x2 = check_img_size(x1+patch_stride_x+margin, 64)
                    y2 = check_img_size(y1+patch_stride_y+margin, 64)
                    patch = img[:, :, y1:y2, x1:x2]
                    pred = model(patch, augment=opt.augment)[0]

                    pred_raw = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                    pred_list.append(pred_raw)
        else:
            x_i = quadrant % patch_num[1]
            y_i = quadrant // patch_num[1]
            x1 = check_img_size(x_i * patch_stride_x, 64)
            y1 = check_img_size(y_i * patch_stride_y, 64)
            x2 = check_img_size(x1+patch_stride_x+margin, 64)
            y2 = check_img_size(y1+patch_stride_y+margin, 64)
            patch = img[:, :, y1:y2, x1:x2]
            pred = model(patch, augment=opt.augment)[0]
            t2 = time_synchronized()
            
            pred_raw = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            pred_prev[quadrant] = pred_raw[0]
            for quadrant_idx in range((patch_num[0]*patch_num[1])):
                pred_list.append(pred_prev[quadrant_idx])
        
        print(pred_list)
        pred_list = tuple(pred_list)
        pred = [torch.cat(pred_list, 0)]

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)            

            clean_im0 = im0.copy()

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            gn_to = torch.tensor(img.shape)[[3, 2, 3, 2]] 
            if len(det):
                scores = det[:, 4]
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape)#.round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                valid_head_list = []
                for xyxy_head in det[det[:, -1] == 1, :4]:
                    valid_head_list.append(xyxy_head.detach().cpu().numpy())
                for xyxy_person in det[det[:, -1] == 0, :4]:
                    xyxy_person_np = xyxy_person.detach().cpu().numpy()
                    is_in_head_all = False
                    for xyxy_head in det[det[:, -1] == 1, :4]:
                        xyxy_head_np = xyxy_head.detach().cpu().numpy()
                        is_in_head = check_head_in_body(xyxy_head_np, xyxy_person_np, 10)
                        if is_in_head:
                            is_in_head_all = True
                    if not is_in_head_all:
                        valid_head_list.append(np.array([xyxy_person_np[0],xyxy_person_np[1],xyxy_person_np[2],(xyxy_person_np[1]*2 + xyxy_person_np[3])/3]))

                for xyxy in valid_head_list:
                    head_point = (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2))
                    im0 = cv2.circle(im0, head_point, 5, (0, 0, 255), -1)                        
                    bbox_num+=1
                    
                tl = 3
                conf = "Head Count: " + str(len(valid_head_list))
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(conf, 0, fontScale=tl / 3, thickness=tf)[0]
                c1 = (0, t_size[1]*2)
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
                cv2.putText(im0, conf, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


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
                        cv2.imwrite(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0])+'_'+str(frame)+'.jpg', im0)
                        if len(det) > 0:
                            cv2.imwrite(os.path.join(save_dir, 'clean_frames', p.name.split('.')[0])+'_'+str(frame)+'_clean.jpg', clean_im0)
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
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps / frame_ratio), (w, h))
                    if opt.save_frame:
                        print(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0]))
                        cv2.imwrite(os.path.join(save_dir, 'vis_frames', p.name.split('.')[0])+'_'+str(frame)+'.jpg', im0)
                        if len(det) > 0:
                            cv2.imwrite(os.path.join(save_dir, 'clean_frames', p.name.split('.')[0])+'_'+str(frame)+'_clean.jpg', clean_im0)
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
    parser.add_argument('--all-quadrant-at-once', action='store_true', help='inference all quadrant at once')
    parser.add_argument('--patch-num', nargs='+', type=int, default=[2,2], help='number of patches y,x')
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
