import argparse
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
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, non_max_suppression_seg, \
    mask_iou, process_semantic_mask
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.torch_utils import select_device, time_synchronized, TracedModel
from collections import defaultdict


#[NOTA]테스트 함수 정의 부분
##############################[C. 테스트 함수 정의]##############################
def test(data,
         weights=None,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         single_cls=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         half_precision=True,
         trace=False,
         v5_metric=False,
         opt_seg=False,
         valid_cls_idx=[]):
         
    #[NOTA]학습해둔 딥러닝 모델을 불러옵니다.
    ##############################[D. 딥러닝 모델 불러오기]##############################
    # Initialize/load model and set device
    set_logging()
    device = select_device(opt.device, batch_size=1)

    # Directories
    save_dir = Path(increment_path(Path('runs/test') / opt.name, exist_ok=False))  # increment run
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
    ##############################[D. 딥러닝 모델 불러오기]##############################
    

    #[NOTA]테스트 데이터를 불러옵니다.
    ##############################[E. 테스트 데이터 불러오기]##############################
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    valid_idx = data.get('valid_idx', None)

    dataloader = create_dataloader(data['val'], imgsz, 1, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{'val'}: '), valid_idx=valid_idx, load_seg=opt_seg)[0]
    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    ##############################[E. 테스트 데이터 불러오기]############################## 
    
    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    ap, ap_class, = [], []
    size_stats = None
    stats = []

    #[NOTA]불러온 테스트 데이터에서 이미지를 하나씩 가져와 딥러닝 테스트.
    ##############################[F. 딥러닝 테스트]##############################
    miou = [[] for _ in range(nc)]
    for batch_i, (img, targets, paths, shapes, masks) in enumerate(tqdm(dataloader, desc=s)):
        if len(valid_cls_idx) > 0:
            valid_target_idx = []
            for t_cls_idx, t_cls in enumerate(targets[:, 1]):
                if t_cls in valid_cls_idx:
                    valid_target_idx.append(t_cls_idx)
            targets = targets[valid_target_idx]
            masks = masks[valid_target_idx]
            
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        masks = masks.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        #[NOTA]딥러닝 모델이 각 이미지에 대해 디텍션을 수행합니다.
        ##############################[G. 디텍션 / 세그멘테이션]##############################
        with torch.no_grad():
            # Run model
            t = time_synchronized()            
            out, train_out = model(img, augment=False)  # inference and training outputs
            t0 += time_synchronized() - t

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

            if opt_seg:
                proto = train_out[1]
        ##############################[G. 디텍션 / 세그멘테이션]##############################


        #[NOTA]디텍션 결과를 저장할 준비를 합니다.
        ##############################[H. 결과 저장 준비]##############################
        # Statistics per image
        for si, pred in enumerate(out):
            if opt_seg:
                seg_pred = process_semantic_mask(proto[si], pred[:, 6:], pred[:, :6], img.shape[2:], upsample=True)
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(valid_cls_idx) > 0:
                valid_pred_idx = []
                for p_cls_idx, p_cls in enumerate(pred[:, -1]):
                    if p_cls in valid_cls_idx:
                        valid_pred_idx.append(p_cls_idx)
                pred = pred[valid_pred_idx]
            
            
            if opt_seg:
                pred_mask = torch.flatten(torch.unsqueeze(seg_pred.gt_(0.5).float() , 0), start_dim=1)
                semantic_gt_mask = torch.zeros((nc, masks.shape[1], masks.shape[2]), dtype=torch.long, device=device)                
                for t_cls_idx, t_cls in enumerate(targets[:, 1]):
                    semantic_gt_mask[int(t_cls)][((masks[t_cls_idx])!=0).bool()] = 1

                gt_mask = torch.flatten(semantic_gt_mask.float(), start_dim=1)
                ious = torch.squeeze(mask_iou(pred_mask, gt_mask), 0)
                max_ious_idx = torch.argmax(ious)
                miou[max_ious_idx].append(ious[max_ious_idx])

            if len(pred) == 0:
                if nl:
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
            ##############################[H. 결과 저장 준비]##############################


            #[NOTA]디텍션 결과와 실제 정답 사이의 IoU(겹침정도)를 계산하여 정답 여부를 판단합니다.
            ##############################[I. IoU 계산]##############################
            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])

                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

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
            ##############################[I. IoU 계산]##############################


    #[NOTA]정답으로 판별된 디텍션 결과들을 모아 mAP를 계산합니다.
    ##############################[K. mAP 계산]##############################
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    ##############################[K. mAP 계산]##############################


    #[NOTA]디텍션 결과를 출력 및 저장합니다.
    ##############################[L. 결과 출력]##############################    
    if opt_seg: # Print Segmentation Results
        for c, iou in enumerate(miou):
            if len(iou) > 0:
                print(names[c] , " : ", sum(iou)/len(iou))
    else:
        # Print Detection Results
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        if nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, 1)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
    ##############################[L. 결과 출력]##############################
##############################[C. 테스트 함수 정의]##############################


#[NOTA]코드를 시작할 때 설정한 값들을 받아 함수로 넘겨주는 부분입니다.
##############################[A. 시작 부분]##############################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--seg', action='store_true', help='Segmentation-Training')
    parser.add_argument('--valid-cls-idx', nargs='+', type=int, default=[], help='labels to include when calculating mAP')
    parser.add_argument('--task', default='det', help='should be one of [det, seg, cls]')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    #[NOTA]넘겨받은 설정값으로 테스트 함수를 시작하는 코드입니다. 테스트 함수 정의 부분으로 이동하여 자세한 테스트 과정을 볼 수 있습니다.
    ##############################[B. 테스트 함수 호출]##############################
    if opt.task in ('train', 'val', 'test'):  # run normally
    test(opt.data,
            opt.weights,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.single_cls,
            opt.verbose,
            save_txt=opt.save_txt | opt.save_hybrid,
            save_hybrid=opt.save_hybrid,
            save_conf=opt.save_conf,
            trace=not opt.no_trace,
            v5_metric=opt.v5_metric,
            opt_seg=opt.seg,
            valid_cls_idx=opt.valid_cls_idx
            )
    ##############################[B. 테스트 함수 호출]##############################
##############################[A. 시작 부분]##############################