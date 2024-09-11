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
            cv2.circle(im0, (int(intersection[0]), int(intersection[1])) , 1, [0,0,255], -1)
    
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

def tracking(sort_tracker, im0, det, min_age, static_lines, roi_polygons, names, colors):
    dets_to_sort = np.empty((0,6))
    
    for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
        dets_to_sort = np.vstack((dets_to_sort, 
                    np.array([x1, y1, x2, y2, conf, detclass])))
    
    # Run SORT
    tracked_dets = sort_tracker.update(dets_to_sort)
    tracks =sort_tracker.getTrackers()

    txt_str = ""
    
    #loop over tracks
    tracking_points_list = []
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

    for tracked_det in tracked_dets:
        tracking_end_point = [(tracked_det[0]+tracked_det[2])/2, (tracked_det[1]+tracked_det[3])/2]
        tracking_end_points.append(tracking_end_point)

    if static_lines is not None:
        intersections_list, counts = count_line_crossing(static_lines, tracking_points_list)
        im0 = draw_intersections(im0, intersections_list, counts, static_lines)

    if roi_polygons is not None:
        in_polygon_points_list, in_polygon_counts = count_in_polygon(im0, roi_polygons, tracking_end_points)
        im0 = draw_in_polygon_points(im0, in_polygon_points_list, in_polygon_counts, roi_polygons)

    # draw boxes for visualization
    if len(tracked_dets)>0:
        bbox_xyxy = tracked_dets[:,:4]
        identities = tracked_dets[:, 8]
        categories = tracked_dets[:, 4]
        draw_boxes(im0, bbox_xyxy, identities, categories, names, colors=colors)
    
    return im0

def detect(save_img=False):
    #.... Initialize SORT ....  
    sort_max_age = -1 # negative age means infinite 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    min_age = 3
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)

    static_lines = [[0, 0, imgsz[1] , imgsz[0]]]
    roi_polygons = [np.array([
        [int(imgsz[1]/3), int(imgsz[0]/3)], 
        [int(imgsz[1]/2), int(2*imgsz[0]/3)], 
        [int(2*imgsz[1]/3), int(imgsz[0]/3)]
        ], dtype=np.int32)]

    names = ['person', 'car']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    #.... detection ....
    #skip#

    #tracking
    source_video = 'cctv006.mp4'
    prediction_examples_source = 'prediction_examples'
    prediction_examples = [torch.load(os.path.join(prediction_examples_source, pred_file)) for pred_file in os.listdir(prediction_examples_source)]
    video = cv2.VideoCapture(source_video) 
    frame_id = -1
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    while(video.isOpened()):
        frame_id = frame_id+1
        ret, im0 = video.read()
        det = prediction_examples[frame_id]
        #im0 : type: numpy, shape: [width, height, 3]
        #det : type: torch.Tensor, shape: [box_num, 6] (6 for [x1, y1, x2, y2, confidence_score, class_id])    
        if ret:
            im0 = tracking(sort_tracker, im0, det, min_age=min_age, static_lines=static_lines, roi_polygons=roi_polygons, names=names, colors=colors)
        elif not ret:
            print("Done")
            break