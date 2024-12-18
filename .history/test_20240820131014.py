import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load, attempt_load_v2, fuse_modules
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, non_max_suppression_seg, \
    mask_iou, masks_iou, process_semantic_mask
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel, intersect_dicts
from models.yolo import Model
import collections
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
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         person_only=False,
         opt_size_division=False,
         opt_seg=False,
         valid_cls_idx=[],
         merge_label=[],
         opt_infinite_names=False,
         qat=False):
    
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
        if qat:
            device = torch.device('cpu')
    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        if qat:
            device = torch.device('cpu')

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        if opt.cfg:
            nc = len(merge_label)
            ckpt_weights = torch.load(opt.weights[0], map_location=device)  # load checkpoint

            if type(ckpt_weights['ema']) is Model:
                state_dict = ckpt_weights['ema'].float().state_dict()  # to FP32
            elif type(ckpt_weights['ema']) is collections.OrderedDict:
                state_dict = ckpt_weights['ema']  # to FP32
            else:
                assert (type(ckpt_weights['ema']) is Model or type(ckpt_weights['ema']) is collections.OrderedDict), "Invalid model types to load"

            ckpt = Model(opt.cfg, ch=3, nc=nc, qat=qat).to(device)  # create#, nm=nm).to(device)  # create
            #if opt.qat:
            #    fuse_modules(ckpt)
            model = attempt_load_v2(ckpt, state_dict, map_location=device)  # load FP32 model
        else:
            model = attempt_load(opt.weights, map_location=device)  # load FP32 model
            if not hasattr(model, 'qat'):
                setattr(model, 'qat', False)
        
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        if len(imgsz) == 2:
            imgsz = [check_img_size(x, gs) for x in imgsz]  # verify imgsz are gs-multiples
            imgsz = tuple(imgsz)
        else:
            imgsz = check_img_size(imgsz[0], gs)  # verify imgsz are gs-multiples
            imgsz = tuple([imgsz, imgsz])
        
        print(imgsz)
        if trace:
            model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and (not qat) and half_precision  # half precision only supported on CUDA
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

    #fuse models
    if qat and not training:
        print(device)
        model = model.cpu()
        #fuse_modules(model)
        
        # The old 'fbgemm' is still available but 'x86' is the recommended default.
        model.qconfig = torch.quantization.get_default_qconfig('x86')

        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        print('Qauntization-Test')

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        
        valid_idx = data.get('valid_idx', None)
        
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, rect=False,
                                       prefix=colorstr(f'{task}: '), valid_idx=valid_idx, load_seg=opt_seg, ratio_maintain=True)[0]
    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    if opt_infinite_names:
        names = {id_: str(id_) for id_ in range(9999)}
    else:
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, ap, ap_class, wandb_images = [], [], [], []
    if opt_size_division:
        size_stats = defaultdict(list)
    else:
        size_stats = None
    stats = []

    if len(merge_label) > 0:
        nc = len(merge_label)
        names = [str(n_num) for n_num in range(len(merge_label))]
    
    miou = [[] for _ in range(nc)]
    for batch_i, (img, targets, paths, shapes, masks) in enumerate(tqdm(dataloader, desc=s)):
        '''
        if len(valid_cls_idx) > 0:
            valid_target_idx = []
            for t_cls_idx, t_cls in enumerate(targets[:, 1]):
                if t_cls in valid_cls_idx:
                    valid_target_idx.append(t_cls_idx)
            targets = targets[valid_target_idx]
            masks = masks[valid_target_idx]
        '''

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

            # Compute loss
            if compute_loss and not opt_seg:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            #if opt_seg:
            #    out = non_max_suppression_seg(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True, nm=32)
            #else:
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

            if not compute_loss and opt_seg:
                proto = train_out[1]

        #print(out)#[torch.size([bboxnum, 6])*32]
        # Statistics per image
        for si, pred in enumerate(out):
            if not compute_loss and opt_seg:
                #masks = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)
                seg_pred = process_semantic_mask(proto[si], pred[:, 6:], pred[:, :6], img.shape[2:], upsample=True)
            labels = targets[targets[:, 0] == si, 1:]            
            #print(labels) # [[  cls,  x,  y,  w,  h], ... ]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            size_division = []
            for label in labels:
                if label[3]*label[4] < 16*16:
                    size_division_ = 'small'
                elif 16*16 <= label[3]*label[4] < 32*32:
                    size_division_ = 'medium'
                else:
                    size_division_ = 'large'
                
                size_division.append(size_division_)
            size_division = np.array(size_division)
            
            '''
            if len(valid_cls_idx) > 0:
                valid_pred_idx = []
                for p_cls_idx, p_cls in enumerate(pred[:, -1]):
                    if p_cls in valid_cls_idx:
                        valid_pred_idx.append(p_cls_idx)
                pred = pred[valid_pred_idx]
            '''         
            
            if not compute_loss and opt_seg: #only bs 1 is possible
                pred_masks_per_cls = torch.zeros((nc, masks.shape[1], masks.shape[2]), dtype=torch.long, device=device)
                semantic_gt_mask = torch.zeros((nc, masks.shape[1], masks.shape[2]), dtype=torch.long, device=device)
                for t_cls_idx, t_cls in enumerate(targets[targets[:, 0] == si, 1]):
                    pred_masks_per_cls[int(t_cls)][seg_pred==t_cls+1] = 1
                    semantic_gt_mask[int(t_cls)][((masks[targets[:, 0] == si][t_cls_idx])!=0).bool()] = 1

                pred_mask = torch.flatten(pred_masks_per_cls.float(), start_dim=1)
                gt_mask = torch.flatten(semantic_gt_mask.float(), start_dim=1)
                ious = torch.squeeze(masks_iou(pred_mask, gt_mask), 0)
                max_ious_idx = torch.argmax(ious)
                miou[max_ious_idx].append(ious[max_ious_idx])

                #vis mask start
                vis_img = cv2.imread(str(path.resolve()))
                image_masks = semantic_gt_mask[-1].detach().cpu().numpy().astype(float)#[label_indexing]
                image_masks = cv2.resize(image_masks, (vis_img.shape[1], vis_img.shape[0]), interpolation = cv2.INTER_NEAREST)
                
                vis_mask = vis_img.copy()
                vis_mask[image_masks!=0] = np.array([50,50,255])
                alpha = 0.5
                vis_img = cv2.addWeighted(vis_img, alpha, vis_mask, 1 - alpha, 0)
                
                image_masks = pred_masks_per_cls[-1].detach().cpu().numpy().astype(float)#[label_indexing]
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
            

            if len(pred) == 0:
                if nl:
                    if opt_size_division:
                        for size_division_ in ['small', 'medium', 'large']:
                            size_stats[size_division_].append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), np.array(tcls)[size_division==size_division_]))
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            
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
            if opt_size_division:
                correct_size_division = {}
                conf_size_division = {}
                for size_division_ in ['small', 'medium', 'large']:                        
                    correct_size_division[size_division_] = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                    conf_size_division[size_division_] = torch.clone(pred[:, 4])
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
                                if opt_size_division:
                                    correct_size_division[size_division[ti[i[j]].detach().cpu().numpy()][0]][pi[j]] = ious[j] > iouv
                                    for size_division__ in ['small', 'medium', 'large']:
                                        if size_division__ != size_division[ti[i[j]].detach().cpu().numpy()][0]:
                                            conf_size_division[size_division__][pi[j]] = 0
                                if len(detected) == nl:  # all targets already located in image
                                    break
                                
            # Append statistics (correct, conf, pcls, tcls)
            if opt_size_division and len(size_division) > 0:
                for size_division_ in ['small', 'medium', 'large']:
                    sort_indices = torch.argsort(conf_size_division[size_division_], descending=True)
                    correct_size_division_ = correct_size_division[size_division_][sort_indices].cpu()
                    conf_size_division_ = conf_size_division[size_division_][sort_indices].cpu()
                    class_size_division_ = pred[:, 5][sort_indices].cpu()
                    #print(size_division_, " : ", conf_size_division_)
                    size_stats[size_division_].append((correct_size_division_, conf_size_division_, class_size_division_,
                            [tcls[i] for i, is_size in enumerate(size_division==size_division_) if is_size]))
            
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))


        # Plot images
        if plots and batch_i < 3:
            if not opt_seg:
                f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
                Thread(target=plot_images, args=(img, targets, paths, f, None, names), daemon=True).start()
                f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
                Thread(target=plot_images, args=(img, output_to_target(out), paths, f, None, names), daemon=True).start()

    # Compute statistics
    if opt_size_division:
        for size_division_ in ['small', 'medium', 'large']:
            print(size_division_+"          Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95")
            size_stats_ = size_stats[size_division_]
            size_stats_ = [np.concatenate(x, 0) for x in zip(*size_stats_)]  # to numpy
            if len(size_stats_) and size_stats_[0].any():
                p, r, ap, f1, ap_class = ap_per_class(*size_stats_, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                nt = np.bincount(size_stats_[3].astype(np.int64), minlength=nc)  # number of targets per class
            else:
                nt = torch.zeros(1)

            # Print results
            pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
            print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

            # Print results per class
            if (verbose or (not training)) and nc > 1 and len(size_stats_):
                for i, c in enumerate(ap_class):
                    print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
        print("all size       Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95")

    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz[0], imgsz[1], batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    if not compute_loss and opt_seg:
        for c, iou in enumerate(miou):
            if len(iou) > 0:
                print(names[c] , " : ", sum(iou)/len(iou))
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
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
    parser.add_argument('--size-division', action='store_true', help='show mAP for small, medium and large objects, respectively')
    parser.add_argument('--seg', action='store_true', help='Segmentation-Training')
    parser.add_argument('--valid-cls-idx', nargs='+', type=int, default=[], help='labels to include when calculating mAP')
    parser.add_argument('--merge-label', type=int, nargs='+', action='append', default=[], help='list of merge label list chunk. --merge-label 0 1 --merge-label 2 3 4')
    parser.add_argument('--infinite-names', action='store_true', help='Do not use saved names in model')
    parser.add_argument('--qat', action='store_true', help='Quantization-Aware-Training')

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
             opt_size_division=opt.size_division,
             opt_seg=opt.seg,
             valid_cls_idx=opt.valid_cls_idx,
             merge_label=opt.merge_label,
             opt_infinite_names=opt.infinite_names,
             qat=opt.qat
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
