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
from sort import Sort

def draw_in_polygon_points(im0, in_polygon_points_list, in_polygon_counts, roi_polygons):
    in_polygons = 0

    fill_image = im0.copy()
    fill_image = cv2.fillPoly(fill_image, pts=roi_polygons, color=(0,255,0))
    alpha = 0.4
    im0 = cv2.addWeighted(fill_image, alpha, im0, 1 - alpha, 0)

    for in_polygon_points in in_polygon_points_list:
        for in_polygon_point in in_polygon_points:
            cv2.circle(im0, (int(in_polygon_point[0]), int(in_polygon_point[1])) , 2, [127,0,255], -1)
    
    for count in in_polygon_counts:
        in_polygons+=count

    tl = 3
    text = "In polygons:"+str(in_polygons)
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
    c1 = (im0.shape[1]-t_size[0], t_size[1])
    c2 = c1[0] + t_size[0], c1[1] - t_size[1]
    cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
    cv2.putText(im0, text, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im0

def count_in_polygon(im0, roi_polygons, tracking_end_points):
    # roi_polygons : [[x1, y1, x2, y2, x3, y3, ...], [x1, y1, x2, y2, x3, y3, ...], ...]
    # tracking_points : [[(x,y), (x,y), ...], ...]
    result_points_list = []
    counts = []
    for roi_polygon in roi_polygons:
        roi_mask = np.zeros(im0.shape[:2])
        roi_mask = cv2.fillPoly(roi_mask, pts=[roi_polygon], color=1)

        result_points = []
        count = 0
        for tracking_end_point in tracking_end_points:
            if roi_mask[int(tracking_end_point[1]), int(tracking_end_point[0])] == 1:
                result_points.append(tracking_end_point)
                count+=1
        result_points_list.append(result_points)
        counts.append(count)
    return result_points_list, counts

def draw_intersections(im0, intersections_list, counts, static_lines):
    crossing_lines = 0
    for static_line in static_lines:
        cv2.line(im0, (int(static_line[0]),
                        int(static_line[1])), 
                        (int(static_line[2]),
                        int(static_line[3])),
                        (0,255,0), thickness=1) 
    for intersections in intersections_list:
        for intersection in intersections:
            cv2.circle(im0, (int(intersection[0]), int(intersection[1])) , 4, [0,0,255], 2)
    
    for count in counts:
        crossing_lines+=count

    tl = 3
    text = "Crossing Lines:"+str(crossing_lines)
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
    c1 = (0, t_size[1])
    c2 = c1[0] + t_size[0], c1[1] - t_size[1]
    cv2.rectangle(im0, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
    cv2.putText(im0, text, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im0

def intersect(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denominator = ((y4 - y3) * (x2 - x1)) - ((x4 - x3) * (y2 - y1))

    if denominator == 0:
        return False

    ua = (((x4 - x3) * (y1 - y3)) - ((y4 - y3) * (x1 - x3))) / denominator
    ub = (((x2 - x1) * (y1 - y3)) - ((y2 - y1) * (x1 - x3))) / denominator

    if ua < 0 or ua > 1 or ub < 0 or ub > 1:
        return False

    intersection_x = x1 + (ua * (x2 - x1))
    intersection_y = y1 + (ua * (y2 - y1))

    return intersection_x, intersection_y

def count_line_crossing(static_lines, tracking_points_list):
    # static_lines : [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    # tracking_points : [[(x,y), (x,y), ...], ...]
    intersections_list = []
    counts = []
    for static_line in static_lines:
        intersections = []
        count = 0
        for tracking_points in tracking_points_list:
            for i, tracking_point in enumerate(tracking_points[:-1]):
                tracking_line = [tracking_point[0], tracking_point[1], tracking_points[i+1][0], tracking_points[i+1][1]]
                intersection = intersect(static_line, tracking_line)
                if intersection:
                    intersections.append(intersection)
                    count+=1
        intersections_list.append(intersections)
        counts.append(count)
    return intersections_list, counts

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

def tracking(sort_tracker, im0, det, min_age, static_lines, previous_intersections, roi_polygons, names, colors):
    #im0 : type: numpy, shape: [width, height, 3]
    #det : type: torch.Tensor, shape: [box_num, 6] (6 for [x1, y1, x2, y2, confidence_score, class_id])   
    dets_to_sort = np.empty((0,6))
    
    for x1,y1,x2,y2,conf,detclass in det:
        dets_to_sort = np.vstack((dets_to_sort, 
                    np.array([x1, y1, x2, y2, conf, detclass])))
    
    # Run SORT
    tracked_dets, dead_trackers = sort_tracker.update(dets_to_sort)
    tracks =sort_tracker.getTrackers()

    #loop over tracks
    tracking_points_list = []
    dead_tracking_points_list = []
    tracking_end_points = []
    for track in tracks:
        if track.age > min_age:
            track_centroids = track.centroidarr[min_age:]
            # color = compute_color_for_labels(id)
            #draw colored tracks
            [cv2.line(im0, (int(track_centroids[i][0]),
                            int(track_centroids[i][1])), 
                            (int(track_centroids[i+1][0]),
                            int(track_centroids[i+1][1])),
                            (255,0,0), thickness=2) 
                            for i,_ in  enumerate(track_centroids) 
                            if i < len(track_centroids)-1 ]            
            tracking_points_list.append(track_centroids)

    #add dead trackers
    for dead_tracker in dead_trackers:
        if dead_tracker.age > min_age:
            track_centroids = dead_tracker.centroidarr[min_age:]
            dead_tracking_points_list.append(track_centroids)

    for tracked_det in tracked_dets:
        #print("tracked_det: ", tracked_det)
        tracking_end_point = [(tracked_det[0]+tracked_det[2])/2, (tracked_det[1]+tracked_det[3])/2]
        tracking_end_points.append(tracking_end_point)

    if static_lines is not None:
        intersections_list, counts = count_line_crossing(static_lines, tracking_points_list)
        dead_intersections_list, dead_counts = count_line_crossing(static_lines, dead_tracking_points_list)
        for i in range(len(static_lines)):
            for dead_intersection in dead_intersections_list[i]:
                previous_intersections[i].append(dead_intersection)
            counts[i]+=len(previous_intersections[i])

        print(counts)
        im0 = draw_intersections(im0, intersections_list, counts, static_lines)
        im0 = draw_intersections(im0, previous_intersections, counts, static_lines)

    if roi_polygons is not None:
        in_polygon_points_list, in_polygon_counts = count_in_polygon(im0, roi_polygons, tracking_end_points)
        im0 = draw_in_polygon_points(im0, in_polygon_points_list, in_polygon_counts, roi_polygons)

    # draw boxes for visualization
    if len(tracked_dets)>0:
        bbox_xyxy = tracked_dets[:,:4]
        identities = tracked_dets[:, 5]
        categories = tracked_dets[:, 4]
        #draw_boxes(im0, bbox_xyxy, identities, categories, names, colors=colors)
    
    return im0, previous_intersections

def detect(save_img=False):
    bbox_num = 0
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    #.... Initialize SORT ....  
    sort_max_age = -1 # negative age means infinite 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    min_age = 5
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)

    static_lines = [[0, 0, imgsz[1] , imgsz[0]]]
    roi_polygons = [np.array([
        [int(imgsz[1]/3), int(imgsz[0]/3)], 
        [int(imgsz[1]/2), int(2*imgsz[0]/3)], 
        [int(2*imgsz[1]/3), int(imgsz[0]/3)]
        ], dtype=np.int32)]
    previous_intersections = [[]]*len(static_lines)
    #......................... 

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
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
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

            clean_im0 = im0.copy()

            if opt.frame_ratio <= 0:
                frame_ratio = fps

            if dataset.mode != 'image' and frame % frame_ratio != 0:
                continue
            else:
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
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    print(frame)
                    np.save("prediction_examples/det_"+str(frame) , det.detach().cpu().numpy())
                    im0, previous_intersections = tracking(sort_tracker, im0, det.detach().cpu().numpy(), min_age=min_age, static_lines=static_lines, previous_intersections=previous_intersections,
                           roi_polygons=roi_polygons, names=names, colors=colors)       

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        
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

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

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
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
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
