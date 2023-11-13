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
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, non_max_suppression_seg, \
    mask_iou, masks_iou, process_semantic_mask
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel
from collections import defaultdict
import cv2


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         person_only=False,
         valid_cls_idx=[]):
    # Initialize/load model and set device

    set_logging()
    device = select_device(opt.device, batch_size=batch_size)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    
    if trace:
        model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    
    valid_idx = data.get('valid_idx', None)

    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '), valid_idx=valid_idx, load_seg=True)[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, ap, ap_class, wandb_images = [], [], [], []
    size_stats = None
    stats = []

    miou = [[] for _ in range(nc)]
    for batch_i, (img, targets, paths, shapes, masks) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        masks = masks.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()            
            out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

            proto = train_out[1]

        #print(out)#[torch.size([bboxnum, 6])*32]
        # Statistics per image
        for si, pred in enumerate(out):
            seg_pred = process_semantic_mask(proto[si], pred[:, 6:], pred[:, :6], img.shape[2:], upsample=True)
            labels = targets[targets[:, 0] == si, 1:]            
            #print(labels) # [[  cls,  x,  y,  w,  h], ... ]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1
        
            pred_masks_per_cls = torch.zeros((2, masks.shape[1], masks.shape[2]), dtype=torch.long, device=device)
            semantic_gt_mask = torch.zeros((2, masks.shape[1], masks.shape[2]), dtype=torch.long, device=device)
            for t_cls_idx, t_cls in enumerate(targets[targets[:, 0] == si, 1]):
                if t_cls == 12:
                    pred_masks_per_cls[0][seg_pred==t_cls+1] = 1
                    semantic_gt_mask[0][((masks[targets[:, 0] == si][t_cls_idx])!=0).bool()] = 1

            pred_mask = torch.flatten(pred_masks_per_cls.float(), start_dim=1)
            gt_mask = torch.flatten(semantic_gt_mask.float(), start_dim=1)
            ious = torch.squeeze(masks_iou(pred_mask, gt_mask), 0)
            max_ious_idx = torch.argmax(ious)
            miou[max_ious_idx].append(ious[max_ious_idx])

            #vis mask start
            vis_img = cv2.imread(str(path.resolve()))
            image_masks = semantic_gt_mask[0].detach().cpu().numpy().astype(float)#[label_indexing]
            image_masks = cv2.resize(image_masks, (vis_img.shape[1], vis_img.shape[0]), interpolation = cv2.INTER_NEAREST)
            
            vis_mask = vis_img.copy()
            vis_mask[image_masks!=0] = np.array([50,50,255])
            alpha = 0.5
            vis_img = cv2.addWeighted(vis_img, alpha, vis_mask, 1 - alpha, 0)
            
            image_masks = pred_masks_per_cls[0].detach().cpu().numpy().astype(float)#[label_indexing]
            image_masks = cv2.resize(image_masks, (vis_img.shape[1], vis_img.shape[0]), interpolation = cv2.INTER_NEAREST)
            
            vis_mask = vis_img.copy()
            vis_mask[image_masks!=0] = np.array([255,50,50])
            alpha = 0.5
            vis_img = cv2.addWeighted(vis_img, alpha, vis_mask, 1 - alpha, 0)

            tl = 2
            vis_txt = str(ious[max_ious_idx])
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(vis_txt, 0, fontScale=tl / 3, thickness=tf)[0]
            c1 = (0, t_size[1]*2)
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(vis_img, c1, c2, (0,0,0), -1, cv2.LINE_AA)  # filled
            cv2.putText(vis_img, vis_txt, (c1[0], c1[1] - 2), 0, tl / 3, [255, 200, 255], thickness=tf, lineType=cv2.LINE_AA)
            cv2.imwrite('test_iou/'+str(path.resolve()).split('/')[-1], vis_img)
            #vis mask end
                
            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred	

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])

                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

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

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    for c, iou in enumerate(miou):
        if len(iou) > 0:
            print(names[c] , " : ", sum(iou)/len(iou))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--person-only', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--xyxy', action='store_true', help='the box label type is xyxy not xywh')
    parser.add_argument('--valid-cls-idx', nargs='+', type=int, default=[], help='labels to include when calculating mAP')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             person_only=opt.person_only,
             valid_cls_idx=opt.valid_cls_idx
             )