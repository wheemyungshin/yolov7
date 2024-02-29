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
from utils.general import check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.torch_utils import select_device, time_synchronized, TracedModel
import cv2

def test(data,
         weights=None,
         imgsz=(256,192),
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         half_precision=True,
         trace=False,
         is_coco=False,
         video_root=None):

    set_logging()
    device = select_device(opt.device, batch_size=1)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    
    if trace:
        model = TracedModel(model, device, imgsz)

    # Half
    #half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    #if half:
    #    model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes

    # Logging
    log_imgs = 0
    # Dataloader
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))  # run once
   
    seen = 0
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    p, r, f1, t0, t1 = 0., 0., 0., 0., 0.
    jdict = []

    # load video
    video_path_list = []
    if os.path.isdir(video_root):
        print("Video Path is Dir.")
        for video_file in os.listdir(video_root):
            video_path = os.path.join(video_root, video_file)
            video_path_list.append(video_path)
    else:
        video_path_list = [video_root]

    # read video
    for video_path in video_path_list:
        print("Loading...:", video_path)
        video = cv2.VideoCapture(video_path)
        #assert video.opened, f'Faild to load video file {video_path}'

        # whether to return heatmap, optional
        return_heatmap = False

        # return the output of some desired layers,
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        print('Running inference...')
        image_id = -1
        while(video.isOpened()):
            image_id = image_id+1
            ret, img = video.read()
            origin_shapes = img.size
            print(origin_shapes)
            if ret:
                if not (imgsz[1] == 0 or imgsz[0] == 0):
                    img = cv2.resize(img, (imgsz[0], imgsz[1]), interpolation=cv2.INTER_AREA)
                
                print("Frame_id: ", str(image_id), end='\r')

                img = torch.unsqueeze(torch.from_numpy(img[:, :, ::-1].copy().transpose(2, 0, 1)).to(device), 0)
                #img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0

                with torch.no_grad():
                    # Run model
                    t = time_synchronized()
                    out, train_out = model(img, augment=augment)  # inference and training outputs
                    t0 += time_synchronized() - t

                    # Run NMS
                    t = time_synchronized()
                    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=[], multi_label=True)
                    t1 += time_synchronized() - t

                # Statistics per image
                for si, pred in enumerate(out):
                    seen += 1

                    if len(pred) == 0:
                        continue

                    # Predictions
                    predn = pred.clone()     
                    scale_coords(img[si].shape[1:], predn[:, :4], origin_shapes[si][0], origin_shapes[si][1])        

                    # Append to pycocotools JSON dictionary
                    if save_json:
                        # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                        box = xyxy2xywh(predn[:, :4])  # xywh
                        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                        for p, b in zip(pred.tolist(), box.tolist()):
                            jdict.append({'image_id': image_id,
                                        'category_id': int(p[5])+1 if is_coco else int(p[5]),
                                        'bbox': [round(x, 3) for x in b],
                                        'score': round(p[4], 5),
                                        'video_path': video_path.split("/")[-1]})
            else:
                break
        print("Done: ", video_path)
        video.release()

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz[0], imgsz[1], 1)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nsaving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

    # Return results
    model.float()  # for training

    print(f"Results saved to {save_dir}")
    return t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--img-size', type=int, default=(256,192), help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')    
    parser.add_argument('--video-root', type=str, help='Video path (video file or dir)')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    test(opt.data,
            opt.weights,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls,
            opt.augment,
            save_txt=opt.save_txt | opt.save_hybrid,
            save_hybrid=opt.save_hybrid,
            save_conf=opt.save_conf,
            trace=not opt.no_trace,
            video_root=opt.video_root
            )

