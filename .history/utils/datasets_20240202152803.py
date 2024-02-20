# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
import json

import pickle
from copy import deepcopy
#from pycocotools import mask as maskUtils
from torchvision.transforms import RandomAffine
from torchvision.utils import save_image
from torchvision.ops import roi_pool, roi_align, ps_roi_pool, ps_roi_align

from utils.general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, pose_xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str
from utils.torch_utils import torch_distributed_zero_first

from collections import defaultdict
import albumentations as A

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(input_img, alpha_c, input_img, 0, gamma_c)
    else:
        buf = input_img.copy()

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(buf, alpha_b, buf, 0, gamma_b)

    return buf
    
def check_boxes_overlap(box1, box2, margin=0):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Check if box1 is to the right of box2
    if x1 > x4 + margin:
        return False

    # Check if box1 is to the left of box2
    if x2 < x3 - margin:
        return False

    # Check if box1 is below box2
    if y1 > y4 + margin:
        return False

    # Check if box1 is above box2
    if y2 < y3 - margin:
        return False
    return True

def gaussian_illumination(img):
    img = img.astype(np.uint8)
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generate a random Gaussian gradient mask for the illumination change
    rows, cols = img.shape[:2]
    kernel_size = np.random.randint((min(rows, cols)//3)*2, min(rows, cols))
    kernel = cv2.getGaussianKernel(192, np.random.randint(45, 55))
    kernel = cv2.resize(kernel, dsize=(kernel_size, kernel_size), interpolation=cv2.INTER_CUBIC)
    mask = kernel @ kernel.T

    # Randomly select the center coordinates of the kernel
    cy = np.random.randint(rows-kernel_size)
    cx = np.random.randint(cols-kernel_size)

    pad_mask = np.zeros((rows, cols))
    pad_mask[cy:cy+kernel_size, cx:cx+kernel_size] = mask

    # Scale and shift the values of the mask to control the range of the illumination change
    if random.random() < 0.5:
        pad_mask = (pad_mask - pad_mask.min()) / (pad_mask.max() - pad_mask.min()) * np.random.randint(180, 250)
    else:
        pad_mask = (pad_mask - pad_mask.min()) / (pad_mask.max() - pad_mask.min()) * np.random.randint(0, 50)

    # Convert the mask to the same data type as the V channel of the HSV image
    pad_mask = pad_mask.astype(hsv.dtype)

    # Add the mask to the V channel of the HSV image
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], pad_mask)

    # Convert the image back to the BGR color space
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr.astype(np.float64)

def gaussuian_filter(size, sigma=1): 
    kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(15)/2)**2+(y-(15)/2)**2))/(2*sigma**2)), (15, 15))
    kernel = cv2.resize(kernel, dsize=(size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    kernel /= np.max(kernel)
    #kernel -= np.mean(kernel)
    return kernel

def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def random_wave(img):
    rows, cols = img.shape[:2]

    # Randomly generate displacement fields
    dx = (np.random.rand(rows, cols) - 0.5) * 1
    #dy = (np.random.rand(rows, cols) - 0.5) * 10

    # Generate grid coordinates
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Add the displacement fields to the grid coordinates
    map_x = (x + np.random.randint(2,6) * np.sin(y / np.random.randint(16,64))).astype(np.float32)
    map_y = (y + np.random.randint(2,6) * np.sin(map_x / np.random.randint(16,64))).astype(np.float32)

    # Warp the image using the generated map
    warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    return warped_img

def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False, ratio_maintain=True,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='', valid_idx=None, pose_data=None, load_seg=False, gray=False):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      ratio_maintain=ratio_maintain,
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix,
                                      valid_idx=valid_idx,
                                      pose_data=pose_data,
                                      load_seg=load_seg,
                                      gray=gray,
                                      merge_label=opt.merge_label)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    if rank != -1:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset, shuffle=True)
        is_shuffle = False
    else:
        sampler = None
        is_shuffle = True
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        shuffle=is_shuffle,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn,)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            url = eval(s) if s.isnumeric() else s
            if 'youtube.com/' in str(url) or 'youtu.be/' in str(url):  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                url = pafy.new(url).getbest(preftype="mp4").url
            cap = cv2.VideoCapture(url)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            if self.fps == 0:
                time.sleep(1 / 90)  # wait time
            else:
                time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

def img2seg_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'segments' + os.sep  # /images/, /labels/ substrings
    seg_list = []
    for x in img_paths:
        seg_path = 'txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) 
        if os.path.exists(seg_path):
            seg_list.append(seg_path)
        else:
            la, lb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
            label_path = 'txt'.join(x.replace(la, lb, 1).rsplit(x.split('.')[-1], 1)) 
            seg_list.append(label_path)
            #print("Using label file (", label_path, ") instead segment (", seg_path, ")")

    return seg_list

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, ratio_maintain=True, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='',valid_idx=None, pose_data=None,
                 load_seg=False, gray=False, merge_label=[]):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.ratio_maintain = ratio_maintain
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        if isinstance(img_size, tuple):
            self.mosaic_border = [-img_size[0] // 2, -img_size[1] // 2]
        else:
            self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.gray = gray
        #self.albumentations = Albumentations() if augment else None
        
        if pose_data is not None:
            if pose_data[0] is not None:
                self.pose_data = defaultdict(list)
                for pose_data_ in pose_data[0]:
                    with open(pose_data_, 'r') as f:
                        train_pose_image_info = json.load(f)['images']
                        image_id_name_match = {}
                        for info_ in train_pose_image_info:
                            image_id_name_match[info_['id']] = info_['file_name']

                    with open(pose_data_, 'r') as f:
                        train_pose = json.load(f)['annotations']
                        for p_dict in train_pose:
                            p_list = np.zeros([17,2])
                            if 'keypoints' in p_dict and 'bbox' in p_dict:
                                for i in range(17):
                                    if p_dict['keypoints'][i*3+2]!= 0:
                                        p_list[i,0] = p_dict['keypoints'][i*3]
                                        p_list[i,1] = p_dict['keypoints'][i*3+1]
                                info_file_name = image_id_name_match[p_dict['image_id']]
                                self.pose_data[info_file_name].append(p_list)

            if pose_data[1] is not None:
                for pose_data_ in pose_data[1]:
                    with open(pose_data_, 'r') as f:
                        test_pose_image_info = json.load(f)['images']
                        image_id_name_match = {}
                        for info_ in test_pose_image_info:
                            image_id_name_match[info_['id']] = info_['file_name']

                    with open(pose_data_, 'r') as f:
                        test_pose = json.load(f)['annotations']
                        for p_dict in test_pose:
                            p_list = np.zeros([17,2])
                            if 'keypoints' in p_dict and 'bbox' in p_dict:
                                for i in range(17):
                                    if p_dict['keypoints'][i*3+2]!= 0:
                                        p_list[i,0] = p_dict['keypoints'][i*3]
                                        p_list[i,1] = p_dict['keypoints'][i*3+1]
                                info_file_name = image_id_name_match[p_dict['image_id']]
                                self.pose_data[info_file_name].append(p_list)
        else:
            self.pose_data = None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        if load_seg:
            self.label_files = img2seg_paths(self.img_files)  # labels
        else:
            self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            #if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
            #    cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix, load_seg=load_seg), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        if load_seg:
            self.label_files = img2seg_paths(cache.keys())
        else:
            self.label_files = img2label_paths(cache.keys())  # update

        if valid_idx is not None:
            print("Filtering non-valid samples")
            new_labels = []
            new_segments = []
            for i, x in enumerate(self.labels):
                new_x = []
                new_seg = []
                seg = self.segments[i]
                for j, x_line in enumerate(x):
                    for id_index, v_id in enumerate(valid_idx):
                        if int(x_line[0]) == v_id:
                            new_x.append(np.array([id_index, x_line[1].copy(), x_line[2].copy(), x_line[3].copy(), x_line[4].copy()]))
                            if len(seg) > j:
                                new_seg.append(seg[j])                                
                            else:
                                new_seg.append(np.array([
                                    [x_line[1]-x_line[3]/2, x_line[2]-x_line[4]/2], 
                                    [x_line[1]+x_line[3]/2, x_line[2]-x_line[4]/2],  
                                    [x_line[1]+x_line[3]/2, x_line[2]+x_line[4]/2],
                                    [x_line[1]-x_line[3]/2, x_line[2]+x_line[4]/2]
                                    ]))
                
                new_labels.append(np.array(new_x))
                new_segments.append(np.array(new_seg))
                    
            self.labels = new_labels
            self.segments = new_segments
            
            self.label_files = [self.label_files[i] for i, x in enumerate(self.labels) if len(x)>0]
            self.img_files = [self.img_files[i] for i, x in enumerate(self.labels) if len(x)>0]
            self.shapes = np.array([self.shapes[i] for i, x in enumerate(self.labels) if len(x)>0])
            self.segments = tuple([self.segments[i] for i, x in enumerate(self.labels) if len(x)>0])
            self.labels = [self.labels[i] for i, x in enumerate(self.labels) if len(x)>0]

        if self.pose_data is not None:
            new_pose_data = []
            for i, img_file in enumerate(self.img_files):
                image_file_name = img_file.split("/")[-1]
                rescale_pose_data = []
                if image_file_name in self.pose_data:
                    for p_data_i, p_data in enumerate(self.pose_data[image_file_name]):
                        if p_data_i < len(self.labels[i]):
                            rescale_pose_data.append(p_data / self.shapes[i])
                else:
                    none_array = np.full((17, 2), None)
                    rescale_pose_data.append(none_array)
                new_pose_data.append(rescale_pose_data)
            self.pose_data = new_pose_data

        if hyp is not None:
            box_margin = hyp.get('box_margin', 1.0)
            for i, x in enumerate(self.labels):
                self.labels[i][:, 3] = x[:, 3]*box_margin
                self.labels[i][:, 4] = x[:, 4]*box_margin
        
        if hyp is not None and hyp.get('render_phone', None) is not None:
            self.phone_imgs = []
            phone_folders = os.listdir(hyp.get('render_phone', None)[0])
            for folder in phone_folders :
                if len(folder.split(".")) == 1 :
                    files = os.listdir(os.path.join(hyp.get('render_phone', None)[0], folder))
                    for file in files :
                        # 너무 수직인 각도면 거를 것
                        if not(file[-6:] == "05.png" or  file[-6:] == "13.png") :
                            phone_img = cv2.imread(os.path.join(hyp.get('render_phone', None)[0], folder + "/" + file), cv2.IMREAD_UNCHANGED)
                            self.phone_imgs.append(phone_img)

        if hyp is not None and hyp.get('render_ciga', None) is not None:
            self.ciga_imgs = []
            ciga_folders = os.listdir(hyp.get('render_ciga', None)[0])
            for folder in ciga_folders :
                if len(folder.split(".")) == 1 :
                    files = os.listdir(os.path.join(hyp.get('render_ciga', None)[0], folder))
                    for file in files :
                        # 너무 수직인 각도면 거를 것
                        ciga_img = cv2.imread(os.path.join(hyp.get('render_ciga', None)[0], folder + "/" + file), cv2.IMREAD_UNCHANGED)
                        self.ciga_imgs.append(ciga_img)

        if hyp is not None and hyp.get('render_hand', None) is not None:
            self.hand_imgs = []
            hand_files = os.listdir(hyp.get('render_hand', None)[0])
            for file in hand_files :
                hand_img = cv2.imread(os.path.join(hyp.get('render_hand', None)[0], file), cv2.IMREAD_UNCHANGED)
                self.hand_imgs.append(hand_img)

        #recursive 하게 동작함. 출발 라벨과 도착 라벨이 겹치는 경우 주의!
        print(merge_label)
        total_merge_label_num = 0
        for merge_label_chunk in merge_label:
            total_merge_label_num+=len(merge_label_chunk)
        
        if len(merge_label) > 0:
            for x in self.labels:
                for merge_label_idx, merge_label_chunk in enumerate(merge_label):
                    x[np.isin(x[:, 0], np.array(merge_label_chunk)), 0] = merge_label_idx - total_merge_label_num
            
            for x in self.labels:
                x[:, 0] += total_merge_label_num

        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride
        
        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix='', load_seg=False):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]       

                        #fix label out of bouonds
                        l_fix = []
                        for x_line in l:
                            line_fixed = [x_line[0]]
                            for x_item in x_line[1:]:
                                line_fixed.append(str(round(max(min(float(x_item),1),0),6)))
                            l_fix.append(line_fixed)
                        l = l_fix

                        if load_seg:  # is segment
                            if any([len(x) > 8 for x in l]):  # is segment
                                classes = np.array([x[0] for x in l], dtype=np.float32)
                                segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                                l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                            else:
                                segments = [np.array([
                                    [float(x_line[1])-float(x_line[3])/2, float(x_line[2])-float(x_line[4])/2], 
                                    [float(x_line[1])+float(x_line[3])/2, float(x_line[2])-float(x_line[4])/2],  
                                    [float(x_line[1])+float(x_line[3])/2, float(x_line[2])+float(x_line[4])/2],
                                    [float(x_line[1])-float(x_line[3])/2, float(x_line[2])+float(x_line[4])/2]
                                    ]) for x_line in l]
                        else:
                            segments = [np.array([
                                    [float(x_line[1])-float(x_line[3])/2, float(x_line[2])-float(x_line[4])/2], 
                                    [float(x_line[1])+float(x_line[3])/2, float(x_line[2])-float(x_line[4])/2],  
                                    [float(x_line[1])+float(x_line[3])/2, float(x_line[2])+float(x_line[4])/2],
                                    [float(x_line[1])-float(x_line[3])/2, float(x_line[2])+float(x_line[4])/2]
                                    ]) for x_line in l]
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        poses = np.array([])
        if mosaic:
            # Load mosaic
            if random.random() < 0.8:
                img, labels, poses, segments = load_mosaic(self, hyp, index)
            else:
                img, labels, poses, segments = load_mosaic9(self, hyp, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                if random.random() < 0.8:
                    img2, labels2, poses2, segments2= load_mosaic(self, hyp, random.randint(0, len(self.labels) - 1))
                else:
                    img2, labels2, poses2, segments2 = load_mosaic9(self, hyp, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
                segments = segments + segments2#np.concatenate((segments, segments2), 0)
                poses = np.concatenate((poses, poses2), 0)

            if hyp is not None and random.random() < hyp.get('face_cut_out', 0):
                for face in labels[labels[:, 0]==1, 1:]:
                    width_temp = face[2] - face[0]
                    height_temp = face[3] - face[1]
                    cutx = face[0]+(1.2*random.random()-0.1)*width_temp
                    cuty = face[1]+(1.2*random.random()-0.1)*height_temp
                    cutw = 0.7*width_temp*random.random()
                    cuth = 0.7*height_temp*random.random()
                    cutx_min = int(min(max(cutx - cutw/2, 0), img.shape[1]))
                    cuty_min = int(min(max(cuty - cuth/2, 0), img.shape[0]))
                    cutx_max = int(min(max(cutx + cutw/2, 0), img.shape[1]))
                    cuty_max = int(min(max(cuty + cuth/2, 0), img.shape[0]))
                    img[cuty_min : cuty_max, cutx_min : cutx_max, :] = random.random()*255
            elif hyp is not None and random.random() < hyp.get('cut_out', 0):
                cutx = (0.6*random.random()+0.2)*img.shape[1]
                cuty = (0.6*random.random()+0.2)*img.shape[0]
                cutw = 0.07*random.random()*img.shape[1]
                cuth = 0.07*random.random()*img.shape[0]
                cutx_min = int(min(max(cutx - cutw/2, 0), img.shape[1]))
                cuty_min = int(min(max(cuty - cuth/2, 0), img.shape[0]))
                cutx_max = int(min(max(cutx + cutw/2, 0), img.shape[1]))
                cuty_max = int(min(max(cuty + cuth/2, 0), img.shape[0]))
                img[cuty_min : cuty_max, cutx_min : cutx_max, :] = random.random()*255

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index, ratio_maintain=self.ratio_maintain)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            segments = self.segments[index].copy()

            labels = self.labels[index].copy()
            
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                segments = [xyn2xy(x, ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]) for x in segments]
                if self.pose_data is not None:
                    poses = self.pose_data[index].copy()
                    poses = np.array([pose_xyn2xy(x, ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]) if x.any() is not None else x for x in poses])
            if hyp is not None:
                for _ in range(hyp.get('max_num_face_cut_out', 0)):
                    if random.random() < hyp.get('face_cut_out', 0):
                        for face in labels[labels[:, 0]==1, 1:]:
                            width_temp = face[2] - face[0]
                            height_temp = face[3] - face[1]
                            cutx = face[0]+(1.2*random.random()-0.1)*width_temp
                            cuty = face[1]+(1.2*random.random()-0.1)*height_temp
                            cutw = 0.8*width_temp*random.random()
                            cuth = 0.8*height_temp*random.random()
                            cutx_min = int(min(max(cutx - cutw/2, 0), w))
                            cuty_min = int(min(max(cuty - cuth/2, 0), h))
                            cutx_max = int(min(max(cutx + cutw/2, 0), w))
                            cuty_max = int(min(max(cuty + cuth/2, 0), h))
                            img[cuty_min : cuty_max, cutx_min : cutx_max, :] = random.random()*255
                    elif hyp is not None and random.random() < hyp.get('cut_out', 0):
                        cutx = (0.6*random.random()+0.2)*img.shape[1]
                        cuty = (0.6*random.random()+0.2)*img.shape[0]
                        cutw = 0.07*random.random()*img.shape[1]
                        cuth = 0.07*random.random()*img.shape[0]
                        cutx_min = int(min(max(cutx - cutw/2, 0), img.shape[1]))
                        cuty_min = int(min(max(cuty - cuth/2, 0), img.shape[0]))
                        cutx_max = int(min(max(cutx + cutw/2, 0), img.shape[1]))
                        cuty_max = int(min(max(cuty + cuth/2, 0), img.shape[0]))
                        img[cuty_min : cuty_max, cutx_min : cutx_max, :] = random.random()*255

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels, segments, poses = random_perspective(img, labels, segments=segments, poses=poses,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
            
            
            #img, labels = self.albumentations(img, labels)

            # Augment colorspace
            if 'contrast' in hyp:
                img = apply_brightness_contrast(img, brightness = 0, contrast = random.random()*(hyp['contrast'][1]-hyp['contrast'][0])+hyp['contrast'][0])
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)
            
            if random.random() < hyp['paste_in']:
                sample_labels, sample_images, sample_masks = [], [], [] 
                while len(sample_labels) < 30:
                    sample_labels_, sample_images_, sample_masks_ = load_samples(self, random.randint(0, len(self.labels) - 1))
                    sample_labels += sample_labels_
                    sample_images += sample_images_
                    sample_masks += sample_masks_
                    if len(sample_labels) == 0:
                        break
                img, labels = pastein(img, labels, sample_labels, sample_images, sample_masks)

            if hyp is not None and random.random() < hyp.get('dark_paste_in', 0):
                sample_images, sample_masks = [], []
                while len(sample_images) < 30:
                    _, sample_images_, sample_masks_ = load_samples(self, random.randint(0, len(self.labels) - 1))
                    sample_images += sample_images_
                    sample_masks += sample_masks_
                    if len(sample_images) == 0:
                        break
                labels = dark_pastein(img, labels, sample_images, sample_masks)

        color_sample = cv2.resize(img, (100,100))
        b = np.mean(color_sample[:, :, 0])
        g = np.mean(color_sample[:, :, 1])
        r = np.mean(color_sample[:, :, 2])
        origin_color_sum = b + g + r
        b = b/origin_color_sum
        g = g/origin_color_sum
        r = r/origin_color_sum

        nL = len(labels)  # number of labels
        if nL:
            shape_before = len(labels)
            #body filtering
            if hyp is not None and hyp.get('body_filter', False):
                body_labels = labels[labels[:, 0]==0]
                face_labels = labels[labels[:, 0]==1]

                in_box_dict = {}
                for body_i, body_label in enumerate(body_labels):
                    for face_i, face_label in enumerate(face_labels):
                        if body_label[1]-1 < face_label[1] < body_label[3]+1 and \
                            body_label[2]-1 < face_label[2] < body_label[4]+1 and \
                            body_label[1]-1 < face_label[3] < body_label[3]+1 and \
                            body_label[2]-1 < face_label[4] < body_label[4]+1:
                            if body_i in in_box_dict:
                                #select the highest face
                                if face_labels[in_box_dict[body_i]][2] > face_label[2]:
                                    in_box_dict[body_i] = face_i
                            else:
                                in_box_dict[body_i] = face_i

                ignore_body_i_list = []
                for body_i, face_i in in_box_dict.items():
                    face_height = face_labels[face_i][4] - face_labels[face_i][2]
                    body_under_face_height = body_labels[body_i][4] - face_labels[face_i][4]
                    if body_under_face_height < face_height*0.6:
                        ignore_body_i_list.append(body_i)

                labels_after_filter = []
                for body_i, body_label in enumerate(body_labels):
                    if body_i not in ignore_body_i_list:
                        labels_after_filter.append(body_label)
                for face_i, face_label in enumerate(face_labels):
                    labels_after_filter.append(face_label)
                labels = np.array(labels_after_filter)
            elif hyp is not None:
                #label min size filtering or scaling
                labels_after_filter = []                
                segments_after_filter = []
                for label_idx, label in enumerate(labels):
                    if (label[3]-label[1])*(label[4]-label[2]) > hyp.get('min_scale_up', 0):#if obj min_size exists
                        labels_after_filter.append(label)
                        segments_after_filter.append(segments[label_idx])
                    else:                        
                        if (label[3]-label[1])*(label[4]-label[2]) > hyp.get('min_size', 0):
                            center_x = int((label[1]+label[3])/2)
                            center_y = int((label[2]+label[4])/2)
                            center_w = label[3]-label[1]
                            center_h = label[4]-label[2]
                            scale_up_ratio = hyp.get('min_scale_up', 0)**0.5 / min(center_w, center_h)
                            scale_up_w = int(center_w * scale_up_ratio / 2)
                            scale_up_h = int(center_h * scale_up_ratio / 2)
                            
                            scale_up_sample = cv2.resize(img[int(label[2]):int(label[4]), int(label[1]):int(label[3]), :], 
                                (scale_up_w*2, scale_up_h*2), interpolation=cv2.INTER_LINEAR)
                            scale_up_x1 = max(center_x-scale_up_w, 0)
                            scale_up_x2 = min(center_x+scale_up_w, img.shape[1])
                            scale_up_y1 = max(center_y-scale_up_h, 0)
                            scale_up_y2 = min(center_y+scale_up_h, img.shape[0])
                            img[scale_up_y1 : scale_up_y2, scale_up_x1 : scale_up_x2, :] = scale_up_sample[:scale_up_y2-scale_up_y1, :scale_up_x2-scale_up_x1, :]
                            new_label = label
                            new_label[1] = scale_up_x1
                            new_label[2] = scale_up_y1
                            new_label[3] = scale_up_x2
                            new_label[4] = scale_up_y2
                            labels_after_filter.append(new_label)
                            segments_after_filter.append(segments[label_idx])
                        else:
                            img[int(label[2]):int(label[4]), int(label[1]):int(label[3]), :] = 0
                labels = np.array(labels_after_filter)
                segments = segments_after_filter
            
            if hyp is not None and hyp.get('piecewise_augment', False):
                transform = A.Compose([
                    A.PiecewiseAffine(always_apply=False, p=1.0, scale=(0.02, 0.03), nb_rows=(3, 3), nb_cols=(3, 3), interpolation=4, mask_interpolation=1, cval=0, cval_mask=0, mode='edge', absolute_scale=0, keypoints_threshold=0.01)
                ])
                if len(labels) > 0 and sum(labels[:, 0]==1) > 0:
                    face_labels = labels[labels[:, 0]==1]
                    for face_label in face_labels :
                        ww = int(face_label[3] - face_label[1])
                        hh = int(face_label[4] - face_label[2])

                        # affine 적용 역역 2배
                        a_x1 = int(face_label[1] - ww/2)
                        a_y1 = int(face_label[2] - hh/2)
                        a_x2 = int(face_label[3] + ww/2)
                        a_y2 = int(face_label[4] + hh/2)

                        # 경계 나가지 않도록 처리
                        a_x1 = min(max(0,a_x1),img.shape[1])
                        a_y1 = min(max(0,a_y1),img.shape[0])
                        a_x2 = min(max(0,a_x2),img.shape[1])
                        a_y2 = min(max(0,a_y2),img.shape[0])

                        # piece wise affine 적용
                        img_crop = img[a_y1:a_y2,a_x1:a_x2]
                        try:
                            transformed = transform(image=img_crop)
                            img_crop = transformed['image']
                            img[a_y1:a_y2,a_x1:a_x2] = img_crop
                        except:
                            print("Invalid transform")

            if hyp is not None and random.random() < hyp.get('check_clothes', [None, 0])[1]:
                if len(labels) > 0:
                    if 1 in labels[:, 0]:#face exists
                        face_label = labels[0]

                        '''
                        check_image = np.zeros([192, 192, 4])
                        check_part_total = random.randint(2, 12)
                        mosaic_patch_size = (img.shape[1]*img.shape[0])**0.5
                        for check_num in range(1, check_part_total):
                            thickness = int((mosaic_patch_size/64) + random.random()*(mosaic_patch_size/32))                    
                            check_color = (random.randint(16, 200)*b*3, random.randint(16, 200)*g*3, random.randint(16, 200)*r*3, random.randint(50, 200))
                            check_x_coord = int((check_num / check_part_total) * check_image.shape[1])
                            check_image = cv2.line(check_image, [check_x_coord+random.randint(-3, 3), 10], [check_x_coord+random.randint(-3, 3), check_image.shape[0]-10], 
                                check_color, thickness, lineType=cv2.LINE_AA)
                        check_part_total = random.randint(2, 6)
                        for check_num in range(1, check_part_total):
                            thickness = int((mosaic_patch_size/64) + random.random()*(mosaic_patch_size/32))                    
                            check_color = (random.randint(16, 200)*b*3, random.randint(16, 200)*g*3, random.randint(16, 200)*r*3, random.randint(50, 200))
                            check_y_coord = int((check_num / check_part_total) * check_image.shape[0])
                            check_image = cv2.line(check_image, [10, check_y_coord+random.randint(-3, 3)], [check_image.shape[1]-10, check_y_coord+random.randint(-3, 3)], 
                                check_color, thickness, lineType=cv2.LINE_AA)
                        print(check_image.dtype)
                        '''
                        
                        check_imgs = os.listdir(hyp.get('check_clothes', [None, 0])[0])      
                        check_filename = check_imgs[random.randint(0, len(check_imgs) - 1)]
                        check_image = cv2.imread(os.path.join(hyp.get('check_clothes', [None, 0])[0], check_filename), cv2.IMREAD_UNCHANGED)
                        if len(check_image.shape) == 3 and check_image.shape[2] == 3 :  
                            check_image_size = random.randint(img.shape[1]/4, img.shape[1]/2)
                            check_image = cv2.resize(check_image, (check_image_size, check_image_size), interpolation=cv2.INTER_LINEAR)

                            check_transform = A.Compose([
                                A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.63, 0.63), shift_limit=(-0.05, 0.05), interpolation=0, border_mode=4, value=(0, 0, 0), mask_value=None),
                                A.ColorJitter(always_apply=False, p=0.5, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                                A.Rotate(always_apply=False, p=1.0, limit=(-30, 30), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False)
                            ])

                            transformed = check_transform(image=check_image)
                            check_image = transformed['image']

                            check_image = cv2.cvtColor(check_image, cv2.COLOR_BGR2RGBA).astype(np.float64)
                            check_image[:, :, 3] = check_image[:, :, 3] * (0.3+random.random()*0.4)
                            
                            check_x1 = int(min(max(face_label[1]-(face_label[3]-face_label[1])*(-0.25+random.random()*1.5), 0), img.shape[1]))
                            check_y1 = int(min(max(face_label[4]+(face_label[4]-face_label[2])*(-0.05+random.random()*0.2), 0), img.shape[0]))
                            check_x2 = int(min(max(face_label[3]+(face_label[3]-face_label[1])*(-0.25+random.random()*1.5), 0), img.shape[1]))                            
                            check_width = check_x2 - check_x1
                            check_y2 = int(min(max(check_y1+check_width, 0), img.shape[0]))

                            check_x1 = random.randint(0, img.shape[1]-check_image.shape[1])
                            check_y1 = random.randint(0, img.shape[0]-check_image.shape[0])
                            check_x2 = check_x1 + check_image.shape[1]
                            check_y2 = check_y1 + check_image.shape[0]
                            check_width = check_image.shape[1]
                            cut_off = max(check_y1 + check_width - img.shape[0], 0)
                            #cut_off = max(check_y2 - check_y1, 0)

                            try:
                                for idx_x in range(check_image.shape[1]) :
                                    for idx_y in range(check_image.shape[0]) :

                                        color_sum = np.sum(check_image[idx_y][idx_x][0:3])

                                        if color_sum > 10 :
                                            check_image[idx_y][idx_x][0] = min(int(color_sum * b),255)
                                            check_image[idx_y][idx_x][1] = min(int(color_sum * g),255)
                                            check_image[idx_y][idx_x][2] = min(int(color_sum * r),255)
                            except:
                                check_image = check_image

                            if check_image.shape[0] > 1 and check_image.shape[1] > 1:
                                check_image = cv2.resize(check_image, (check_width, check_width), interpolation=cv2.INTER_LINEAR)
                                check_image = random_wave(check_image)

                                if cut_off > 0:
                                    check_image = check_image[:-cut_off, :, :]

                                if random.randint(0,1) == 0 :
                                    check_image = cv2.flip(check_image, 1)                                
                                        

                                if check_y2 - check_y1 > 1 and check_x2 - check_x1 > 1:
                                    img_crop = img[check_y1:check_y2, check_x1:check_x2]
                                    #img_crop = cv2.resize(img_crop, (check_image.shape[1], check_image.shape[0]), interpolation=cv2.INTER_LINEAR)
                                    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2RGBA)

                                    # Pillow 에서 Alpha Blending
                                    check_image_pillow = Image.fromarray(check_image.astype(np.uint8))
                                    img_crop_pillow = Image.fromarray(img_crop)
                                    blended_pillow = Image.alpha_composite(img_crop_pillow, check_image_pillow)
                                    blended_img=np.array(blended_pillow)  

                                    # 원본 이미지에 다시 합치기
                                    blended_img = cv2.cvtColor(blended_img, cv2.COLOR_RGBA2RGB)
                                    img[check_y1:check_y2, check_x1:check_x2] = blended_img

            if hyp is not None and random.random() < hyp.get('fakeseatbelt3', [None, 0])[1]:
                if len(labels) > 0:
                    if 1 in labels[:, 0] and 0 not in labels[:, 0] and len(labels[:, 0])==1:#face exists, seatbelt does not exist
                        seatbelt_filename = None
                        if random.random() < 0.5:
                            seatbelt_imgs = os.listdir(hyp.get('fakeseatbelt3', [None, 0])[0])      
                            seatbelt_filename = seatbelt_imgs[random.randint(0, len(seatbelt_imgs) - 1)]
                            seatbelt_img = cv2.imread(os.path.join(hyp.get('fakeseatbelt3', [None, 0])[0], seatbelt_filename), cv2.IMREAD_UNCHANGED)                            
                            end_x = seatbelt_img.shape[1]
                            end_y = seatbelt_img.shape[0]
                            if seatbelt_img.shape[0] > 256:
                                if random.random() < hyp.get('obstacle', 0):
                                    remove_top_y = random.randint(20, 60)
                                    seatbelt_img[:int(remove_top_y), : , :] = 0
                                if random.random() < hyp.get('obstacle', 0):
                                    remove_bottom_y = random.randint(30, 100)
                                    seatbelt_img[int(seatbelt_img.shape[0]-remove_bottom_y): , : , :] = 0
                        else:
                            color_element = random.randint(16, 100)
                            alpha_element = random.randint(50, 200)
                            mosaic_patch_size = (img.shape[1]*img.shape[0])**0.5
                            thickness = int((mosaic_patch_size/16) + random.random()*(mosaic_patch_size/16))
                            semi_x = random.randint(50, 80)
                            semi_y = random.randint(80, 110)
                            end_x = random.randint(112, 180)
                            end_y = random.randint(128, 192)
                            seatbelt_img = np.ones([192, 192, 4])
                            seatbelt_mask = np.zeros([192, 192, 4])
                            if random.random() < 0.5:
                                seatbelt_mask = cv2.line(seatbelt_mask, [0, 0], [semi_x, semi_y], 
                                    (1, 1, 1, 1), thickness, lineType=cv2.LINE_AA) 
                            else:
                                seatbelt_mask = cv2.line(seatbelt_mask, [random.randint(20, 45), random.randint(30, 70)], [semi_x, semi_y], 
                                    (1, 1, 1, 1), thickness, lineType=cv2.LINE_AA) 

                            seatbelt_mask = cv2.line(seatbelt_mask, [semi_x, semi_y], [end_x, end_y], 
                                    (1, 1, 1, 1) , thickness, lineType=cv2.LINE_AA)
                            seatbelt_mask = random_wave(seatbelt_mask)
                            seatbelt_mask = seatbelt_mask.astype(bool)

                            seatbelt_img[:, :, 0] = color_element*b*3
                            seatbelt_img[:, :, 1] = color_element*g*3
                            seatbelt_img[:, :, 2] = color_element*r*3
                            seatbelt_img[:, :, 3] = alpha_element
                            seatbelt_img[: ,:, :3] = gaussian_illumination(seatbelt_img[: ,:, :3])

                            seatbelt_img = np.where(seatbelt_mask, seatbelt_img, [0, 0, 0, 0])
                        
                        #obstacle
                        if random.random() < hyp.get('obstacle', 0):
                            obstacle_mask = np.zeros([seatbelt_img.shape[0], seatbelt_img.shape[1]])
                            mosaic_patch_size = (img.shape[1]*img.shape[0])**0.5
                            obstacle_thickness = int((mosaic_patch_size/8) + random.random()*(mosaic_patch_size/8))
                            obstacle_mask = cv2.line(obstacle_mask, [random.randint(int(seatbelt_img.shape[1]/2), seatbelt_img.shape[1]), random.randint(0, int(seatbelt_img.shape[0]/2))], 
                                    [random.randint(0, int(seatbelt_img.shape[1]/2)), random.randint(int(seatbelt_img.shape[0]/2), seatbelt_img.shape[0])], 
                                    (1), obstacle_thickness, lineType=cv2.LINE_AA) 
                            seatbelt_img[obstacle_mask==1, :] = 0

                        face_label = labels[0]
                        if int(face_label[4]) < img.shape[0]*0.8:
                            random_x_transpose = (random.random()-0.5)*(face_label[3]-face_label[1])
                            seat_x1_range = min(max(face_label[1]-(face_label[3]-face_label[1])+random_x_transpose, 0), img.shape[1])
                            seat_y1_range = min(max(face_label[4]-(face_label[4]-face_label[2])*random.random()*0.2, 0), img.shape[0])
                            seat_x2_range = min(max(face_label[3]+(face_label[3]-face_label[1])*(1.1)+random_x_transpose, 0), img.shape[1])
                            if face_label[2] < 1:                                
                                seat_y2_range = min(max(face_label[4]+(face_label[3]-face_label[1])*(1+random.random()), 0), img.shape[0])
                            else:
                                seat_y2_range = min(max(face_label[4]+(face_label[4]-face_label[2])*(1+random.random()), 0), img.shape[0])

                            if seatbelt_filename is not None and (seatbelt_filename.startswith('03') or seatbelt_filename.startswith('04')):
                                seat_x1_start = int(min(max(face_label[1]-(face_label[3]-face_label[1])*1.5+random_x_transpose, 0), img.shape[1]))
                                seat_y1_start = int(seat_y1_range)
                                seat_x2_start = int(min(max(face_label[3]+(face_label[3]-face_label[1])*1.5+random_x_transpose, 0), img.shape[1]))
                                seat_y2_start = int(seat_y2_range)
                            else:
                                seat_x1_start = int(min(max(face_label[1]-(face_label[3]-face_label[1])*(0.25+random.random()*0.5)+random_x_transpose, 0), img.shape[1]))
                                seat_y1_start = int(min(max(random.randint(int(face_label[2]), int(seat_y1_range)), 0), img.shape[0]))
                                seat_x2_start = int(min(max(face_label[3]+(face_label[3]-face_label[1])*(-0.25+random.random())+random_x_transpose, 0), img.shape[1]))
                                seat_y2_start = int(seat_y2_range)

                            color_element = 32+int(random.random()*128)
                            mosaic_patch_size = (img.shape[1]*img.shape[0])**0.5
                            thickness = int((mosaic_patch_size/16) + random.random()*(mosaic_patch_size/16))

                            x1 = int(seat_x1_range)
                            y1 = int(seat_y1_range)
                            x2 = min(max(int(seat_x1_range+(seat_x2_range-seat_x1_range)*(end_x/seatbelt_img.shape[1])), 0), img.shape[1])      
                            y2 = min(max(int(seat_y1_range+(seat_y2_range-seat_y1_range)*(end_y/seatbelt_img.shape[0])), 0), img.shape[0])     
                            
                            if (seat_x2_start-seat_x1_start) > 0 and (seat_y2_start-seat_y1_start) > 0 :
                                seatbelt_img = cv2.resize(seatbelt_img, ((seat_x2_start-seat_x1_start), (seat_y2_start-seat_y1_start)), interpolation=cv2.INTER_LINEAR)

                                try:
                                    for idx_x in range(seatbelt_img.shape[1]) :
                                        for idx_y in range(seatbelt_img.shape[0]) :

                                            color_sum = np.sum(seatbelt_img[idx_y][idx_x][0:3])

                                            if color_sum > 10 :
                                                seatbelt_img[idx_y][idx_x][0] = min(int(color_sum * b),255)
                                                seatbelt_img[idx_y][idx_x][1] = min(int(color_sum * g),255)
                                                seatbelt_img[idx_y][idx_x][2] = min(int(color_sum * r),255)
                                except:
                                    seatbelt_img = seatbelt_img

                                if random.randint(0,1) == 0 :
                                    seatbelt_img = cv2.flip(seatbelt_img, 1)
                                    seatbet_center_x = (seat_x1_start+seat_x2_start)/2
                                    x1 = seatbet_center_x + (seatbet_center_x - x1)
                                    x2 = seatbet_center_x + (seatbet_center_x - x2)

                                img_crop = img[seat_y1_start:seat_y2_start, seat_x1_start:seat_x2_start]
                                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2RGBA)

                                # Pillow 에서 Alpha Blending
                                seatbelt_img_pillow = Image.fromarray(seatbelt_img.astype(np.uint8))
                                img_crop_pillow = Image.fromarray(img_crop)
                                blended_pillow = Image.alpha_composite(img_crop_pillow, seatbelt_img_pillow)
                                blended_img=np.array(blended_pillow)  

                                # 원본 이미지에 다시 합치기
                                blended_img = cv2.cvtColor(blended_img, cv2.COLOR_RGBA2RGB)
                                img[seat_y1_start:seat_y2_start, seat_x1_start:seat_x2_start] = blended_img

                                labels = np.append(labels, np.array([[0, x1, y1, x2, y2]]), axis=0)  
                                segments.append(np.array([
                                    [x1, y1], 
                                    [x2, y1],  
                                    [x2, y2],
                                    [x1, y2]
                                    ]))
        
        ciga_colors = [[0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], 
            [255,100,100], [100,255,100], [100,100,255], [100,255,255], [255,255,100], [255,100,255], [100,255,255],
            [255,255,200], [255,200,255], [200,255,255], [150,150,150], [75,75,75]]
            
        if hyp is not None and hyp.get('cellphone_translation', None) is not None:
            nL = len(labels)  # number of labels
            if nL:
                affine_transfomer = RandomAffine(degrees=(-180, 180), translate=(0.05, 0.15), scale=(0.9, 1.1))

                labels_after_filter = []
                segments_after_filter = []
                
                ciga_masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=1)
                ciga_outer_masks = polygons2lines(img.shape[:2], segments, color=1, downsample_ratio=1, thickness=150)
                #print("segments: ", len(segments))
                #print("ciga_masks: ", ciga_masks.shape)
                #print("labels: ", labels.shape)
                for ciga_idx, ciga_label in enumerate(labels):#enumerate(labels[labels[:, 0]==2]):
                    cutout_random_percent1 = random.random()
                    cutout_random_percent2 = random.random()
                    if ciga_label[0]==2 or ciga_label[0]==3:
                        if cutout_random_percent1 < hyp.get('cellphone_translation', 0):
                            ciga_color = ciga_colors[random.randint(0,len(ciga_colors)-1)]
                            ciga_color[0] = min(max(ciga_color[0] + random.randint(-30, 30), 0), 255)
                            ciga_color[1] = min(max(ciga_color[1] + random.randint(-30, 30), 0), 255)
                            ciga_color[2] = min(max(ciga_color[2] + random.randint(-30, 30), 0), 255)
                            ciga_patch = np.zeros_like(img)
                            ciga_patch[ciga_masks[ciga_idx] != 0] = img[ciga_masks[ciga_idx] != 0]

                            affine_ciga_patch = torch.squeeze(affine_transfomer(torch.from_numpy(ciga_patch).permute(2, 0, 1).unsqueeze(0))).permute(1, 2, 0).numpy()
                            
                            new_ciga_label = ciga_label.copy()
                            non_zero_indices = np.nonzero(affine_ciga_patch)
                            if len(non_zero_indices[0]) > 0 and len(non_zero_indices[1]) > 0:
                                min_x = np.min(non_zero_indices[1])
                                min_y = np.min(non_zero_indices[0])
                                max_x = np.max(non_zero_indices[1])
                                max_y = np.max(non_zero_indices[0])
                                new_ciga_label[1] = min_x
                                new_ciga_label[2] = min_y
                                new_ciga_label[3] = max_x
                                new_ciga_label[4] = max_y
                                if ciga_label[0]==3 and max_x-min_x > 16 and max_y-min_y > 16 and random.random() < hyp.get('render_hand', ['', 0.0])[1]:
                                    hand_img = self.hand_imgs[random.randint(0, len(self.hand_imgs) - 1)]
                                    hand_img = cv2.resize(hand_img, (max_x-min_x, max_y-min_y), interpolation=cv2.INTER_LINEAR)

                                    try:
                                        for idx_x in range(hand_img.shape[1]) :
                                            for idx_y in range(hand_img.shape[0]) :

                                                color_sum = np.sum(hand_img[idx_y][idx_x][0:3])

                                                if color_sum > 10 :
                                                    hand_img[idx_y][idx_x][0] = min(int(color_sum * b),255)
                                                    hand_img[idx_y][idx_x][1] = min(int(color_sum * g),255)
                                                    hand_img[idx_y][idx_x][2] = min(int(color_sum * r),255)
                                    except:
                                        hand_img = hand_img
                                        
                                    img[ciga_masks[ciga_idx] != 0] = ciga_color
                                    img[min_y:max_y, min_x:max_x][hand_img>10] = hand_img[hand_img>10]
                                else:
                                    img[ciga_masks[ciga_idx] != 0] = ciga_color
                                img[affine_ciga_patch != 0] = affine_ciga_patch[affine_ciga_patch != 0]
                                
                                if max_x-min_x > 4 and max_y-min_y > 4:
                                    affine_ciga_patch[affine_ciga_patch != 0] = 1
                                    labels_after_filter.append(new_ciga_label)
                                    segments_after_filter.append(affine_ciga_patch)
                        else:
                            labels_after_filter.append(ciga_label)
                            segments_after_filter.append(segments[ciga_idx])
                    else:
                        labels_after_filter.append(ciga_label)
                        segments_after_filter.append(segments[ciga_idx])

                labels = np.array(labels_after_filter)
                segments = segments_after_filter

        if hyp is not None:
            random_percentage = random.random()
            
            if random_percentage < hyp.get('render_ciga', ['', 0.0])[1]:
                if len(labels) > 0:
                    num_of_ciga_img = [1,2]
                    num_of_ciga = random.randint(num_of_ciga_img[0], num_of_ciga_img[1])
                    for idx in range(num_of_ciga) :
                        ciga_img = self.ciga_imgs[random.randint(0, len(self.ciga_imgs) - 1)]
                        ciga_img = cv2.resize(ciga_img, None, fx=0.075+random.random()*0.05, fy=0.075+random.random()*0.05, interpolation=cv2.INTER_LINEAR)
                        if random.random() > 0.5 :
                            ciga_img = ciga_img[:,::-1,:]
                        if random.random() > 0.5 :
                            ciga_img = ciga_img[::-1,:,:]
                        if random.random() > 0.5 :
                            ciga_img = cv2.rotate(ciga_img, cv2.ROTATE_90_CLOCKWISE)
                        
                        if img.shape[1]-ciga_img.shape[1] > 0 and img.shape[0]-ciga_img.shape[0] > 0:
                            gray = ciga_img[:,:,3]
                            th, threshed = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
                            morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
                            cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                            cnt = sorted(cnts, key=cv2.contourArea)[-1]
                            p_x1,p_y1,p_w,p_h = cv2.boundingRect(cnt)

                            try:
                                for idx_x in range(ciga_img.shape[1]) :
                                    for idx_y in range(ciga_img.shape[0]) :

                                        color_sum = np.sum(ciga_img[idx_y][idx_x][0:3])

                                        if color_sum > 10 :
                                            ciga_img[idx_y][idx_x][0] = min(int(color_sum * b),255)
                                            ciga_img[idx_y][idx_x][1] = min(int(color_sum * g),255)
                                            ciga_img[idx_y][idx_x][2] = min(int(color_sum * r),255)
                            except:
                                ciga_img = ciga_img

                            # ciga 위치 랜덤하게 지정
                            ciga_img_position_x = random.randint(0, img.shape[1]-ciga_img.shape[1])
                            ciga_img_position_y = random.randint(0, img.shape[0]-ciga_img.shape[0])

                            img_crop = img[ciga_img_position_y:ciga_img_position_y+ciga_img.shape[0], ciga_img_position_x:ciga_img_position_x+ciga_img.shape[1]]
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2BGRA)                            
                            
                            ciga_img_pillow = Image.fromarray(ciga_img)
                            img_crop_pillow = Image.fromarray(img_crop)
                            blended_pillow = Image.alpha_composite(img_crop_pillow, ciga_img_pillow)
                            blended_img=np.array(blended_pillow)  

                            blended_img = cv2.cvtColor(blended_img, cv2.COLOR_RGBA2RGB)
                            img[ciga_img_position_y:ciga_img_position_y+ciga_img.shape[0], ciga_img_position_x:ciga_img_position_x+ciga_img.shape[1]] = blended_img

                            min_x = p_x1 + ciga_img_position_x
                            min_y = p_y1 + ciga_img_position_y
                            max_x = p_x1 + p_w + ciga_img_position_x
                            max_y = p_y1 + p_h + ciga_img_position_y
                            new_label = np.array([[2, min_x, min_y, max_x, max_y]])
                            new_segment = np.array([
                                [min_x, min_y], 
                                [max_x, min_y],  
                                [max_x, max_y],
                                [min_x, max_y]
                                ])
                            
                            labels = np.append(labels, new_label, axis=0)  
                            segments.append(new_segment)

            #else : if there is ciga rendering, no phone rendering
            elif random_percentage > 1-hyp.get('render_phone', ['', 0.0])[1]:
                if len(labels) > 0:
                    num_of_phone_img = [1,2]
                    num_of_phone = random.randint(num_of_phone_img[0], num_of_phone_img[1])
                    for idx in range(num_of_phone) :
                        phone_img = self.phone_imgs[random.randint(0, len(self.phone_imgs) - 1)]
                        phone_img = cv2.resize(phone_img, None, fx=0.1+random.random()*0.1, fy=0.1+random.random()*0.1, interpolation=cv2.INTER_LINEAR)
                        if random.random() > 0.5 :
                            phone_img = phone_img[:,::-1,:]
                        if random.random() > 0.5 :
                            phone_img = phone_img[::-1,:,:]
                        if random.random() > 0.5 :
                            phone_img = cv2.rotate(phone_img, cv2.ROTATE_90_CLOCKWISE)
                        
                        if img.shape[1]-phone_img.shape[1] > 0 and img.shape[0]-phone_img.shape[0] > 0:
                            gray = phone_img[:,:,3]
                            th, threshed = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
                            morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
                            cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                            cnt = sorted(cnts, key=cv2.contourArea)[-1]
                            p_x1,p_y1,p_w,p_h = cv2.boundingRect(cnt)

                            try:
                                for idx_x in range(phone_img.shape[1]) :
                                    for idx_y in range(phone_img.shape[0]) :

                                        color_sum = np.sum(phone_img[idx_y][idx_x][0:3])

                                        if color_sum > 10 :
                                            phone_img[idx_y][idx_x][0] = min(int(color_sum * b),255)
                                            phone_img[idx_y][idx_x][1] = min(int(color_sum * g),255)
                                            phone_img[idx_y][idx_x][2] = min(int(color_sum * r),255)
                            except:
                                phone_img = phone_img

                            # phone 위치 랜덤하게 지정
                            phone_img_position_x = random.randint(0, img.shape[1]-phone_img.shape[1])
                            phone_img_position_y = random.randint(0, img.shape[0]-phone_img.shape[0])

                            img_crop = img[phone_img_position_y:phone_img_position_y+phone_img.shape[0], phone_img_position_x:phone_img_position_x+phone_img.shape[1]]
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2BGRA)                            
                            
                            phone_img_pillow = Image.fromarray(phone_img)
                            img_crop_pillow = Image.fromarray(img_crop)
                            blended_pillow = Image.alpha_composite(img_crop_pillow, phone_img_pillow)
                            blended_img=np.array(blended_pillow)  

                            blended_img = cv2.cvtColor(blended_img, cv2.COLOR_RGBA2RGB)
                            img[phone_img_position_y:phone_img_position_y+phone_img.shape[0], phone_img_position_x:phone_img_position_x+phone_img.shape[1]] = blended_img

                            min_x = p_x1 + phone_img_position_x
                            min_y = p_y1 + phone_img_position_y
                            max_x = p_x1 + p_w + phone_img_position_x
                            max_y = p_y1 + p_h + phone_img_position_y
                            new_label = np.array([[3, min_x, min_y, max_x, max_y]])
                            new_segment = np.array([
                                [min_x, min_y], 
                                [max_x, min_y],  
                                [max_x, max_y],
                                [min_x, max_y]
                                ])
                            
                            labels = np.append(labels, new_label, axis=0)  
                            segments.append(new_segment)

        if hyp is not None and random.random() < hyp.get('render_hand', ['', 0.0])[1]:
            num_of_hand_img = [1,3]
            num_of_hand = random.randint(num_of_hand_img[0], num_of_hand_img[1])
            for idx in range(num_of_hand) :
                hand_img = self.hand_imgs[random.randint(0, len(self.hand_imgs) - 1)]
                hand_img = cv2.resize(hand_img, None, fx=0.12+random.random()*0.23, fy=0.12+random.random()*0.23, interpolation=cv2.INTER_LINEAR)

                try:
                    for idx_x in range(hand_img.shape[1]) :
                        for idx_y in range(hand_img.shape[0]) :

                            color_sum = np.sum(hand_img[idx_y][idx_x][0:3])

                            if color_sum > 10 :
                                hand_img[idx_y][idx_x][0] = min(int(color_sum * b),255)
                                hand_img[idx_y][idx_x][1] = min(int(color_sum * g),255)
                                hand_img[idx_y][idx_x][2] = min(int(color_sum * r),255)
                except:
                    hand_img = hand_img

                # hand 위치 랜덤하게 지정
                if img.shape[1]-hand_img.shape[1] > 0 and img.shape[0]-hand_img.shape[0] > 0:
                    hand_img_position_x = random.randint(0, img.shape[1]-hand_img.shape[1])
                    hand_img_position_y = random.randint(0, img.shape[0]-hand_img.shape[0])
                    
                    is_invalid_position = False
                    for ciga_label in labels:
                        if ciga_label[0]==2 or ciga_label[0]==3:
                            if check_boxes_overlap([hand_img_position_x, hand_img_position_y, hand_img_position_x+hand_img.shape[1], hand_img_position_y+hand_img.shape[0]],
                                [ciga_label[1], ciga_label[2], ciga_label[3], ciga_label[4]], 0):
                                is_invalid_position=True
                                
                    if not is_invalid_position:
                        img_crop = img[hand_img_position_y:hand_img_position_y+hand_img.shape[0], hand_img_position_x:hand_img_position_x+hand_img.shape[1]]
                        img_crop[hand_img>40] = hand_img[hand_img>40]
                        img[hand_img_position_y:hand_img_position_y+hand_img.shape[0], hand_img_position_x:hand_img_position_x+hand_img.shape[1]] = img_crop


        if hyp is not None and (hyp.get('ciga_cutout', None) is not None or hyp.get('cellphone_cutout', None) is not None):
            nL = len(labels)  # number of labels
            if nL:
                labels_after_filter = []
                segments_after_filter = []
                
                ciga_masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=1)
                ciga_outer_masks = polygons2lines(img.shape[:2], segments, color=1, downsample_ratio=1, thickness=150)
                #print("segments: ", len(segments))
                #print("ciga_masks: ", ciga_masks.shape)
                #print("labels: ", labels.shape)
                for ciga_idx, ciga_label in enumerate(labels):#enumerate(labels[labels[:, 0]==2]):
                    cutout_random_percent1 = random.random()
                    cutout_random_percent2 = random.random()
                    '''
                    if ciga_label[0]==2:
                        if cutout_random_percent1 < hyp.get('ciga_cutout', 0):
                            ciga_color = ciga_colors[random.randint(0,len(ciga_colors)-1)]
                            img[ciga_masks[ciga_idx] != 0] = ciga_color
                        elif cutout_random_percent1 > 1 - hyp.get('ciga_outer_cutout', 0):
                            ciga_color = ciga_colors[random.randint(0,len(ciga_colors)-1)]
                            ciga_outer_masks[ciga_idx][np.sum(ciga_masks[labels[:, 0]==2], axis=0) != 0] = 0
                            img[ciga_outer_masks[ciga_idx] != 0] = ciga_color                        
                            labels_after_filter.append(ciga_label)
                            segments_after_filter.append(segments[ciga_idx])
                        else:
                            labels_after_filter.append(ciga_label)
                            segments_after_filter.append(segments[ciga_idx])

                    elif ciga_label[0]==3:
                        if cutout_random_percent2 < hyp.get('cellphone_cutout', 0):
                            ciga_color = ciga_colors[random.randint(0,len(ciga_colors)-1)]
                            img[ciga_masks[ciga_idx] != 0] = ciga_color
                        elif cutout_random_percent2 > 1 - hyp.get('cellphone_outer_cutout', 0):
                            ciga_color = ciga_colors[random.randint(0,len(ciga_colors)-1)]
                            ciga_outer_masks[ciga_idx][np.sum(ciga_masks[labels[:, 0]==3], axis=0) != 0] = 0
                            ciga_outer_masks[ciga_idx][np.sum(ciga_masks[labels[:, 0]==4], axis=0) != 0] = 1
                            img[ciga_outer_masks[ciga_idx] != 0] = ciga_color                        
                            labels_after_filter.append(ciga_label)
                            segments_after_filter.append(segments[ciga_idx])
                        else:
                            labels_after_filter.append(ciga_label)
                            segments_after_filter.append(segments[ciga_idx])
                    elif ciga_label[0]==4:
                        if not (cutout_random_percent2 > 1 - hyp.get('cellphone_outer_cutout', 0)):#hand cutout togehter with cellphone
                            labels_after_filter.append(ciga_label)
                            segments_after_filter.append(segments[ciga_idx])
                    '''
                    if ciga_label[0]==2:
                        if cutout_random_percent2 < hyp.get('cellphone_cutout', 0):
                            ciga_color = ciga_colors[random.randint(0,len(ciga_colors)-1)]
                            img[ciga_masks[ciga_idx] != 0] = ciga_color
                        elif cutout_random_percent2 > 1 - hyp.get('cellphone_outer_cutout', 0):
                            ciga_color = ciga_colors[random.randint(0,len(ciga_colors)-1)]
                            ciga_outer_masks[ciga_idx][np.sum(ciga_masks[labels[:, 0]==2], axis=0) != 0] = 0
                            #ciga_outer_masks[ciga_idx][np.sum(ciga_masks[labels[:, 0]==3], axis=0) != 0] = 1
                            img[ciga_outer_masks[ciga_idx] != 0] = ciga_color                        
                            labels_after_filter.append(ciga_label)
                            segments_after_filter.append(segments[ciga_idx])
                        else:
                            labels_after_filter.append(ciga_label)
                            segments_after_filter.append(segments[ciga_idx])
                    else:
                        labels_after_filter.append(ciga_label)
                        segments_after_filter.append(segments[ciga_idx])
                labels = np.array(labels_after_filter)
                segments = segments_after_filter
                        
                        
        if hyp is not None and hyp.get('render_fire', None) is not None:
            nL = len(labels)  # number of labels
            if nL:
                fire_imgs = []
                for fire_source_dir in hyp.get('render_fire', None)[0]:
                    for fire_img_name in os.listdir(fire_source_dir):
                        fire_img_path = os.path.join(fire_source_dir, fire_img_name)
                        fire_imgs.append(fire_img_path)
                if len(labels[labels[:, 0]==2]) > 0:
                    is_valid_ciga = labels[:, 0]==2
                    valid_idx = 0
                    while valid_idx < len(is_valid_ciga):
                        is_valid = is_valid_ciga[valid_idx]
                        if not is_valid:
                            if valid_idx-1 >= 0:
                                is_valid_ciga[valid_idx-1] = False
                            if valid_idx+1 < len(is_valid_ciga):
                                if is_valid_ciga[valid_idx+1]==True:
                                    is_valid_ciga[valid_idx+1] = False
                                    valid_idx+=2
                                else:
                                    valid_idx+=1
                            else:
                                valid_idx+=1
                        else:
                            valid_idx+=1

                    for idx, ciga_label in enumerate(labels[is_valid_ciga]) :
                        min_ciga_size = 16
                        if min(ciga_label[3]-ciga_label[1], ciga_label[4]-ciga_label[2]) > min_ciga_size and random.random() < 0.5:
                            fire_size = max(8, random.randint(int(min(ciga_label[3]-ciga_label[1], ciga_label[4]-ciga_label[2])/5), int(min(ciga_label[3]-ciga_label[1], ciga_label[4]-ciga_label[2])/3)))
                            fire_img = cv2.imread(fire_imgs[random.randint(0, len(fire_imgs) - 1)], cv2.IMREAD_UNCHANGED)
                            fire_img = cv2.resize(fire_img, (fire_size, fire_size), cv2.INTER_CUBIC)

                            for idx_x in range(fire_img.shape[1]) :
                                for idx_y in range(fire_img.shape[0]) :
                                    color_sum = np.sum(fire_img[idx_y][idx_x][0:3])
                                    if color_sum > 10 :
                                        fire_img[idx_y][idx_x][0] = min(int(color_sum * b),255)
                                        fire_img[idx_y][idx_x][1] = min(int(color_sum * g),255)
                                        fire_img[idx_y][idx_x][2] = min(int(color_sum * r),255)
                            
                            # 번쩍임 추가
                            fire_img = cv2.convertScaleAbs(fire_img, alpha=random.random()*0.1+1.0, beta=random.randint(10, 20))

                            # 회전 여부 랜덤
                            if random.randint(0,1) == 0 :
                                fire_img = cv2.flip(fire_img, 0)
                            if random.randint(0,1) == 0 :
                                fire_img = cv2.flip(fire_img, 1)

                            # fire 위치 랜덤하게 지정
                            if (ciga_label[3]-ciga_label[1])*2 < (ciga_label[4]-ciga_label[2]): # h is longer
                                fire_img_position_x = random.randint(max(int(ciga_label[1])-int(min_ciga_size/2), 0), 
                                    min(int(ciga_label[3])-fire_img.shape[1]+int(min_ciga_size/2), img.shape[1]-fire_img.shape[1]))
                                fire_img_position_y = random.choice([int(ciga_label[2]), int(ciga_label[4])-fire_img.shape[0]])
                            elif (ciga_label[4]-ciga_label[2])*2 < (ciga_label[3]-ciga_label[1]): # w is longer
                                fire_img_position_x = random.choice([int(ciga_label[1]), int(ciga_label[3])-fire_img.shape[1]])
                                fire_img_position_y = random.randint(max(int(ciga_label[2])-int(min_ciga_size/2), 0), 
                                    min(int(ciga_label[4])-fire_img.shape[0]+int(min_ciga_size/2), img.shape[0]-fire_img.shape[0]))
                            else:
                                fire_img_position_x = random.randint(max(int(ciga_label[1])-int(min_ciga_size/2), 0), 
                                    min(int(ciga_label[3])-fire_img.shape[1]+int(min_ciga_size/2), img.shape[1]-fire_img.shape[1]))
                                fire_img_position_y = random.randint(max(int(ciga_label[2])-int(min_ciga_size/2), 0), 
                                    min(int(ciga_label[4])-fire_img.shape[0]+int(min_ciga_size/2), img.shape[0]-fire_img.shape[0]))
                            img_crop = img[fire_img_position_y:fire_img_position_y+fire_img.shape[0], fire_img_position_x:fire_img_position_x+fire_img.shape[1]]
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2RGBA)

                            # Pillow 에서 Alpha Blending
                            fire_img_pillow = Image.fromarray(fire_img)
                            img_crop_pillow = Image.fromarray(img_crop)
                            blended_pillow = Image.alpha_composite(img_crop_pillow, fire_img_pillow)
                            blended_img=np.array(blended_pillow)  

                            # 원본 이미지에 다시 합치기
                            blended_img = cv2.cvtColor(blended_img, cv2.COLOR_RGBA2RGB)
                            img[fire_img_position_y:fire_img_position_y+fire_img.shape[0], fire_img_position_x:fire_img_position_x+fire_img.shape[1]] = blended_img

                            # 빛 번짐 감안하여 이미지에 margin 을 주었기 때문에 0.13 씩 상하좌우 빼줍니다.
                            labels = np.append(labels, [[hyp.get('render_fire', None)[1], 
                                fire_img_position_x + int(fire_img.shape[1]* 0.13), fire_img_position_y + int(fire_img.shape[0]* 0.13), 
                                fire_img_position_x + fire_img.shape[1] -int(fire_img.shape[1]*0.13), fire_img_position_y + fire_img.shape[0] -int(fire_img.shape[0]*0.13)]], axis=0)


                    # retry for edge
                    for idx, edge_label in enumerate(labels[labels[:, 0]==3]) :
                        if random.random() < 0.5:
                            edge_width = edge_label[3]-edge_label[1]
                            edge_height = edge_label[4]-edge_label[2]
                            fire_size = int(min(edge_width, edge_height))
                            fire_img = cv2.imread(fire_imgs[random.randint(0, len(fire_imgs) - 1)], cv2.IMREAD_UNCHANGED)
                            
                            fire_img = cv2.resize(fire_img, (fire_size, fire_size), cv2.INTER_CUBIC)

                            for idx_x in range(fire_img.shape[1]) :
                                for idx_y in range(fire_img.shape[0]) :
                                    color_sum = np.sum(fire_img[idx_y][idx_x][0:3])
                                    if color_sum > 10 :
                                        fire_img[idx_y][idx_x][0] = min(int(color_sum * b),255)
                                        fire_img[idx_y][idx_x][1] = min(int(color_sum * g),255)
                                        fire_img[idx_y][idx_x][2] = min(int(color_sum * r),255)
                            
                            # 번쩍임 추가
                            fire_img = cv2.convertScaleAbs(fire_img, alpha=random.random()*0.1+1.0, beta=random.randint(10, 20))

                            # 회전 여부 랜덤
                            if random.randint(0,1) == 0 :
                                fire_img = cv2.flip(fire_img, 0)
                            if random.randint(0,1) == 0 :
                                fire_img = cv2.flip(fire_img, 1)

                            if edge_width > edge_height:
                                fire_img_position_x = int(edge_label[1]+(edge_width-edge_height)/2)
                                fire_img_position_y = int(edge_label[2])
                            else:
                                fire_img_position_x = int(edge_label[1])
                                fire_img_position_y = int(edge_label[2]+(edge_height-edge_width)/2)
                                     
                            img_crop = img[fire_img_position_y:fire_img_position_y+fire_img.shape[0],
                                     fire_img_position_x:fire_img_position_x+fire_img.shape[1]]
                            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2RGBA)

                            # Pillow 에서 Alpha Blending
                            fire_img_pillow = Image.fromarray(fire_img)
                            img_crop_pillow = Image.fromarray(img_crop)
                            blended_pillow = Image.alpha_composite(img_crop_pillow, fire_img_pillow)
                            blended_img=np.array(blended_pillow)  

                            # 원본 이미지에 다시 합치기
                            blended_img = cv2.cvtColor(blended_img, cv2.COLOR_RGBA2RGB)
                            img[fire_img_position_y:fire_img_position_y+fire_img.shape[0], fire_img_position_x:fire_img_position_x+fire_img.shape[1]] = blended_img

                            # 빛 번짐 감안하여 이미지에 margin 을 주었기 때문에 0.13 씩 상하좌우 빼줍니다.
                            labels = np.append(labels, [[hyp.get('render_fire', None)[1], 
                                fire_img_position_x + int(fire_img.shape[1]* 0.13), fire_img_position_y + int(fire_img.shape[0]* 0.13), 
                                fire_img_position_x + fire_img.shape[1] -int(fire_img.shape[1]*0.13), fire_img_position_y + fire_img.shape[0] -int(fire_img.shape[0]*0.13)]], axis=0)


        if hyp is not None and hyp.get('glitter_for_burn', 0) > 0:
            nL = len(labels)  # number of labels
            if nL:
                if len(labels[labels[:, 0]==1]) > 0:
                    for idx, burn_label in enumerate(labels) :
                        if burn_label[0]==1 and random.random() < hyp.get('glitter_for_burn', 0):
                            burn_center_x = (burn_label[1] + burn_label[3]) / 2
                            burn_center_y = (burn_label[2] + burn_label[4]) / 2
                            burn_width = burn_label[3] - burn_label[1]
                            burn_height = burn_label[4] - burn_label[2]
                            crop_x1 = int(burn_center_x - burn_width*(random.random()*0.3+0.7)/2)
                            crop_y1 = int(burn_center_y - burn_height*(random.random()*0.3+0.7)/2)
                            crop_x2 = int(burn_center_x + burn_width*(random.random()*0.3+0.7)/2)
                            crop_y2 = int(burn_center_y + burn_height*(random.random()*0.3+0.7)/2)
                            burn_crop = img[crop_y1:crop_y2,crop_x1:crop_x2].copy()
                            burn_crop[:,:,0] = burn_crop[:,:,0] * (random.random()*0.08+0.77)
                            burn_crop[:,:,1] = burn_crop[:,:,1] * (random.random()*0.08+0.77)
                            burn_crop = cv2.convertScaleAbs(burn_crop, alpha=random.random()*0.5+1.0, beta=random.randint(40, 60))

                            if random.random() < 0.5:
                                gaus_filter = gaussuian_filter((crop_y2-crop_y1, crop_x2-crop_x1), sigma=5)
                                random_mask = np.random.random((crop_y2-crop_y1, crop_x2-crop_x1)) * gaus_filter
                                img[crop_y1:crop_y2,crop_x1:crop_x2][random_mask>0.5] = burn_crop[random_mask>0.5]
                            else:
                                gaus_filter1 = gaussuian_filter((crop_y2-crop_y1, crop_x2-crop_x1), sigma=5)
                                random_mask = gaus_filter1
                                if int((crop_y2-crop_y1)/2) > 1 and int((crop_x2-crop_x1)/2) > 1:
                                    gaus_filter2 = gaussuian_filter((int((crop_y2-crop_y1)/2), int((crop_x2-crop_x1)/2)), sigma=5)
                                    random_mask[int((crop_y2-crop_y1)/4):int((crop_y2-crop_y1)/4+gaus_filter2.shape[0]), 
                                        int((crop_x2-crop_x1)/4):int((crop_x2-crop_x1)/4+gaus_filter2.shape[1])] = 1 - gaus_filter2                            
                                img[crop_y1:crop_y2,crop_x1:crop_x2, 0] = (1-random_mask)*img[crop_y1:crop_y2,crop_x1:crop_x2, 0] + random_mask*burn_crop[:,:,0]
                                img[crop_y1:crop_y2,crop_x1:crop_x2, 1] = (1-random_mask)*img[crop_y1:crop_y2,crop_x1:crop_x2, 1] + random_mask*burn_crop[:,:,1]
                                img[crop_y1:crop_y2,crop_x1:crop_x2, 2] = (1-random_mask)*img[crop_y1:crop_y2,crop_x1:crop_x2, 2] + random_mask*burn_crop[:,:,2]
                            labels[idx][0] = 0

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

            segments = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=1)#self.downsample_ratio)

        segments = (torch.from_numpy(segments) if nL else torch.zeros([0, img.shape[0], img.shape[1]]))
        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
                    segments = torch.flip(segments, dims=[1])
                if self.pose_data is not None and len(poses) > 0:
                    for poses_i in range(len(poses)):
                        for poses_j in range(len(poses[poses_i])):
                            if poses[poses_i][poses_j].any() is not None and not np.isnan(poses[poses_i, poses_j, 1]):
                                poses[poses_i, poses_j, 1] = img.shape[0] - poses[poses_i, poses_j, 1]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
                    segments = torch.flip(segments, dims=[2])
                if self.pose_data is not None and len(poses) > 0:
                    for poses_i in range(len(poses)):
                        for poses_j in range(len(poses[poses_i])):
                            if poses[poses_i][poses_j].any() is not None and not np.isnan(poses[poses_i, poses_j, 0]):
                                poses[poses_i, poses_j, 0] = img.shape[1] - poses[poses_i, poses_j, 0]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)        
        segments_out = segments[:nL]
        
        #Visualize pose joints
        #print(poses.shape)#((4 or 9), 17, 2)
        '''
        img = img.copy()
        if self.pose_data is not None and len(poses) > 0:
            for pose in poses:
                for p in pose:
                    if p.any() is not None and not np.isnan(p[0]) and not np.isnan(p[1]):
                        img = cv2.circle(img, (int(p[0]), int(p[1])), 9, (255, 0, 0), -1)
        '''

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

        img = np.ascontiguousarray(img)
        
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, segments_out

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, segments = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        
            #print(l.shape)
        #print(len(label))
                   
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, torch.cat(segments, 0)

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes, segments = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4, segments4 = [], [], path[:n], shapes[:n], []

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                segment = F.interpolate(segments[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(segments[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                segment = torch.cat((torch.cat((segments[i], segments[i + 1]), 1), torch.cat((segments[i + 2], segments[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            segments4.append(segment)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4, torch.cat(segments4, 0)


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index, ratio_maintain=True):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        if self.gray:
            img[:,:,1] = img[:,:,0]
            img[:,:,2] = img[:,:,0]
        h0, w0 = img.shape[:2]  # orig hw
        if ratio_maintain:
            if isinstance(self.img_size, tuple):
                r = self.img_size[1] / max(h0, w0)  # resize image to img_size
                if r != 1:  # always resize down, only resize up if training with augmentation
                    interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            else:
                r = self.img_size / max(h0, w0)  # resize image to img_size
                if r != 1:  # always resize down, only resize up if training with augmentation
                    interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
        else:
            if isinstance(self.img_size, tuple):
                img = cv2.resize(img, (int(self.img_size[1]), int(self.img_size[0])), interpolation=cv2.INTER_LINEAR)
            else:
                r = self.img_size / max(h0, w0)  # resize image to img_size
                if r != 1:  # always resize down, only resize up if training with augmentation
                    interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    if isinstance(vgain, float):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    else:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, 0] + 1  # random gains
        r[2]=r[2]+(random.random()*(vgain[1]-vgain[0])+vgain[0])
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def adjust_gamma(img, gamma=1.0):
    dtype = img.dtype  # uint8 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(dtype)
    return cv2.LUT(img, table)

def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def load_mosaic(self, hyp, index):
    # loads images in a 4-mosaic

    labels4, segments4, poses4 = [], [], []
    if isinstance(self.img_size, tuple):
        ys = self.img_size[0]
        xs = self.img_size[1]
    else:
        ys = self.img_size
        xs = self.img_size
    yc, xc = [int(random.uniform(-self.mosaic_border[0], 2 * ys + self.mosaic_border[0])), \
            int(random.uniform(-self.mosaic_border[1], 2 * xs + self.mosaic_border[1]))]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((ys * 2, xs * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, xs * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(ys * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, xs * 2), min(ys * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b] # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()

        if self.pose_data is not None:
            poses = self.pose_data[index].copy()
        if labels.size:
            #print("SEGMENTS: ", segments)

            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
    
            if self.pose_data is not None:
                poses = [pose_xyn2xy(x, w, h, padw, padh) if x.any() is not None else x for x in poses]

            if hyp is not None and random.random() < hyp.get('fakeseatbelt3', [None, 0])[1]:
                if len(labels) > 0:
                    if 1 in labels[:, 0] and 0 not in labels[:, 0] and len(labels[:, 0])==1:#face exists, seatbelt does not exist
                        seatbelt_filename = None
                        if random.random() < 0.5:
                            seatbelt_imgs = os.listdir(hyp.get('fakeseatbelt3', [None, 0])[0])      
                            seatbelt_filename = seatbelt_imgs[random.randint(0, len(seatbelt_imgs) - 1)]
                            seatbelt_img = cv2.imread(os.path.join(hyp.get('fakeseatbelt3', [None, 0])[0], seatbelt_filename), cv2.IMREAD_UNCHANGED)                            
                            end_x = seatbelt_img.shape[1]
                            end_y = seatbelt_img.shape[0]
                            if seatbelt_img.shape[0] > 256:
                                if random.random() < hyp.get('obstacle', 0):
                                    remove_top_y = random.randint(20, 60)
                                    seatbelt_img[:int(remove_top_y), : , :] = 0
                                if random.random() < hyp.get('obstacle', 0):
                                    remove_bottom_y = random.randint(30, 100)
                                    seatbelt_img[int(seatbelt_img.shape[0]-remove_bottom_y): , : , :] = 0
                        else:
                            color_element = random.randint(16, 100)
                            alpha_element = random.randint(50, 200)
                            mosaic_patch_size = (img.shape[1]*img.shape[0])**0.5
                            thickness = int((mosaic_patch_size/16) + random.random()*(mosaic_patch_size/16))
                            semi_x = random.randint(50, 80)
                            semi_y = random.randint(80, 110)
                            end_x = random.randint(112, 180)
                            end_y = random.randint(128, 192)
                            seatbelt_img = np.ones([192, 192, 4])
                            seatbelt_mask = np.zeros([192, 192, 4])
                            if random.random() < 0.5:
                                seatbelt_mask = cv2.line(seatbelt_mask, [0, 0], [semi_x, semi_y], 
                                    (1, 1, 1, 1), thickness, lineType=cv2.LINE_AA) 
                            else:
                                seatbelt_mask = cv2.line(seatbelt_mask, [random.randint(20, 45), random.randint(30, 70)], [semi_x, semi_y], 
                                    (1, 1, 1, 1), thickness, lineType=cv2.LINE_AA) 

                            seatbelt_mask = cv2.line(seatbelt_mask, [semi_x, semi_y], [end_x, end_y], 
                                    (1, 1, 1, 1) , thickness, lineType=cv2.LINE_AA)
                            seatbelt_mask = random_wave(seatbelt_mask)
                            seatbelt_mask = seatbelt_mask.astype(bool)

                            seatbelt_img[:, :, 0] = color_element*b*3
                            seatbelt_img[:, :, 1] = color_element*g*3
                            seatbelt_img[:, :, 2] = color_element*r*3
                            seatbelt_img[:, :, 3] = alpha_element
                            seatbelt_img[: ,:, :3] = gaussian_illumination(seatbelt_img[: ,:, :3])

                            seatbelt_img = np.where(seatbelt_mask, seatbelt_img, [0, 0, 0, 0])
                        

                        #obstacle
                        if random.random() < hyp.get('obstacle', 0):
                            obstacle_mask = np.zeros([seatbelt_img.shape[0], seatbelt_img.shape[1]])
                            mosaic_patch_size = (img.shape[1]*img.shape[0])**0.5
                            obstacle_thickness = int((mosaic_patch_size/8) + random.random()*(mosaic_patch_size/8))
                            obstacle_mask = cv2.line(obstacle_mask, [random.randint(int(seatbelt_img.shape[1]/2), seatbelt_img.shape[1]), random.randint(0, int(seatbelt_img.shape[0]/2))], 
                                    [random.randint(0, int(seatbelt_img.shape[1]/2)), random.randint(int(seatbelt_img.shape[0]/2), seatbelt_img.shape[0])], 
                                    (1), obstacle_thickness, lineType=cv2.LINE_AA) 
                            seatbelt_img[obstacle_mask==1, :] = 0


                        face_label = labels[0]
                        if int(face_label[4]) < img.shape[0]*0.8:
                            random_x_transpose = (random.random()-0.5)*(face_label[3]-face_label[1])
                            seat_x1_range = min(max(face_label[1]-(face_label[3]-face_label[1])+random_x_transpose, 0), img.shape[1])
                            seat_y1_range = min(max(face_label[4]-(face_label[4]-face_label[2])*random.random()*0.2, 0), img.shape[0])
                            seat_x2_range = min(max(face_label[3]+(face_label[3]-face_label[1])*(1.1)+random_x_transpose, 0), img.shape[1])
                            seat_y2_range = min(max(face_label[4]+(face_label[4]-face_label[2])*(1+random.random()), 0), img.shape[0])

                            if seatbelt_filename is not None and (seatbelt_filename.startswith('03') or seatbelt_filename.startswith('04')):
                                seat_x1_start = int(min(max(face_label[1]-(face_label[3]-face_label[1])*1.5+random_x_transpose, 0), img.shape[1]))
                                seat_y1_start = int(seat_y1_range)
                                seat_x2_start = int(min(max(face_label[3]+(face_label[3]-face_label[1])*1.5+random_x_transpose, 0), img.shape[1]))
                                seat_y2_start = int(seat_y2_range)
                            else:
                                seat_x1_start = int(min(max(face_label[1]-(face_label[3]-face_label[1])*(0.25+random.random()*0.5)+random_x_transpose, 0), img.shape[1]))
                                seat_y1_start = int(min(max(random.randint(int(face_label[2]), int(seat_y1_range)), 0), img.shape[0]))
                                seat_x2_start = int(min(max(face_label[3]+(face_label[3]-face_label[1])*(-0.25+random.random())+random_x_transpose, 0), img.shape[1]))
                                seat_y2_start = int(seat_y2_range)

                            color_element = 32+int(random.random()*128)
                            mosaic_patch_size = (img.shape[1]*img.shape[0])**0.5
                            thickness = int((mosaic_patch_size/16) + random.random()*(mosaic_patch_size/16))

                            x1 = int(seat_x1_range)
                            y1 = int(seat_y1_range)
                            x2 = min(max(int(seat_x1_range+(seat_x2_range-seat_x1_range)*(end_x/seatbelt_img.shape[1])), 0), img.shape[1])      
                            y2 = min(max(int(seat_y1_range+(seat_y2_range-seat_y1_range)*(end_y/seatbelt_img.shape[0])), 0), img.shape[0])     
                            
                            if (seat_x2_start-seat_x1_start) > 0 and (seat_y2_start-seat_y1_start) > 0 :
                                seatbelt_img = cv2.resize(seatbelt_img, ((seat_x2_start-seat_x1_start), (seat_y2_start-seat_y1_start)), interpolation=cv2.INTER_LINEAR)

                                try:
                                    for idx_x in range(seatbelt_img.shape[1]) :
                                        for idx_y in range(seatbelt_img.shape[0]) :

                                            color_sum = np.sum(seatbelt_img[idx_y][idx_x][0:3])

                                            if color_sum > 10 :
                                                seatbelt_img[idx_y][idx_x][0] = min(int(color_sum * b),255)
                                                seatbelt_img[idx_y][idx_x][1] = min(int(color_sum * g),255)
                                                seatbelt_img[idx_y][idx_x][2] = min(int(color_sum * r),255)
                                except:
                                    seatbelt_img = seatbelt_img

                                if random.randint(0,1) == 0 :
                                    seatbelt_img = cv2.flip(seatbelt_img, 1)
                                    seatbet_center_x = (seat_x1_start+seat_x2_start)/2
                                    x1 = seatbet_center_x + (seatbet_center_x - x1)
                                    x2 = seatbet_center_x + (seatbet_center_x - x2)

                                img_crop = img[seat_y1_start:seat_y2_start, seat_x1_start:seat_x2_start]
                                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2RGBA)

                                # Pillow 에서 Alpha Blending
                                seatbelt_img_pillow = Image.fromarray(seatbelt_img.astype(np.uint8))
                                img_crop_pillow = Image.fromarray(img_crop)
                                blended_pillow = Image.alpha_composite(img_crop_pillow, seatbelt_img_pillow)
                                blended_img=np.array(blended_pillow)  

                                # 원본 이미지에 다시 합치기
                                blended_img = cv2.cvtColor(blended_img, cv2.COLOR_RGBA2RGB)
                                img[seat_y1_start:seat_y2_start, seat_x1_start:seat_x2_start] = blended_img

                                labels = np.append(labels, np.array([[0, x1, y1, x2, y2]]), axis=0) 
                                segments.append(np.array([
                                    [x1, y1], 
                                    [x2, y1],  
                                    [x2, y2],
                                    [x1, y2]
                                    ]))

        labels4.append(labels)
        segments4.extend(segments)
        if self.pose_data is not None:
            poses4.extend(poses)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * max(xs, ys), out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    poses4 = np.array(poses4)
    if self.pose_data is not None:
        for x in poses4:
            if x.any() is not None:
                np.clip(x, 0, 2 * max(xs, ys), out=x)
            
    # Augment
    #img4, labels4, segments4 = remove_background(img4, labels4, segments4)
    #sample_segments(img4, labels4, segments4, probability=self.hyp['copy_paste'])
    #("Labels4: ", labels4.shape)#(n,5)
    #("Segments4: ", segments4.shape)#(n,s,2)
    #("Pose4: ", pose_data4.shape)#(n,17,2)

    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, probability=self.hyp['copy_paste'])
    img4, labels4, segments4, poses4 = random_perspective(img4, labels4, segments4, poses4,
                                        degrees=self.hyp['degrees'],
                                        translate=self.hyp['translate'],
                                        scale=self.hyp['scale'],
                                        shear=self.hyp['shear'],
                                        perspective=self.hyp['perspective'],
                                        border=self.mosaic_border)  # border to remove


    return img4, labels4, poses4, segments4



def load_mosaic9(self, hyp, index):
    # loads images in a 9-mosaic

    labels9, segments9, poses9 = [], [], []
    if isinstance(self.img_size, tuple):
        ys = self.img_size[0]
        xs = self.img_size[1]
    else:
        ys = self.img_size
        xs = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((ys * 3, xs * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = xs, ys, xs + w, ys + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = xs, ys - h, xs + w, ys
        elif i == 2:  # top right
            c = xs + wp, ys - h, xs + wp + w, ys
        elif i == 3:  # right
            c = xs + w0, ys, xs + w0 + w, ys + h
        elif i == 4:  # bottom right
            c = xs + w0, ys + hp, xs + w0 + w, ys + hp + h
        elif i == 5:  # bottom
            c = xs + w0 - w, ys + h0, xs + w0, ys + h0 + h
        elif i == 6:  # bottom left
            c = xs + w0 - wp - w, ys + h0, xs + w0 - wp, ys + h0 + h
        elif i == 7:  # left
            c = xs - w, ys + h0 - h, xs, ys + h0
        elif i == 8:  # top left
            c = xs - w, ys + h0 - hp - h, xs, ys + h0 - hp

        padx, pady = c[:2]
        x1a, y1a, x2a, y2a = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if self.pose_data is not None:
            poses = self.pose_data[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            if self.pose_data is not None:
                poses = [pose_xyn2xy(x, w, h, padx, pady) if x.any() is not None else x for x in poses]

            if hyp is not None and random.random() < hyp.get('fakeseatbelt3', [None, 0])[1]:
                if len(labels) > 0:
                    if 1 in labels[:, 0] and 0 not in labels[:, 0] and len(labels[:, 0])==1:#face exists, seatbelt does not exist                    
                        seatbelt_filename = None
                        if random.random() < 0.5:
                            seatbelt_imgs = os.listdir(hyp.get('fakeseatbelt3', [None, 0])[0])      
                            seatbelt_filename = seatbelt_imgs[random.randint(0, len(seatbelt_imgs) - 1)]
                            seatbelt_img = cv2.imread(os.path.join(hyp.get('fakeseatbelt3', [None, 0])[0], seatbelt_filename), cv2.IMREAD_UNCHANGED)                            
                            end_x = seatbelt_img.shape[1]
                            end_y = seatbelt_img.shape[0]
                            if seatbelt_img.shape[0] > 256:
                                if random.random() < hyp.get('obstacle', 0):
                                    remove_top_y = random.randint(20, 60)
                                    seatbelt_img[:int(remove_top_y), : , :] = 0
                                if random.random() < hyp.get('obstacle', 0):
                                    remove_bottom_y = random.randint(30, 100)
                                    seatbelt_img[int(seatbelt_img.shape[0]-remove_bottom_y): , : , :] = 0
                        else:
                            color_element = random.randint(16, 100)
                            alpha_element = random.randint(50, 200)
                            mosaic_patch_size = (img.shape[1]*img.shape[0])**0.5
                            thickness = int((mosaic_patch_size/16) + random.random()*(mosaic_patch_size/16))
                            semi_x = random.randint(50, 80)
                            semi_y = random.randint(80, 110)
                            end_x = random.randint(112, 180)
                            end_y = random.randint(128, 192)
                            seatbelt_img = np.ones([192, 192, 4])
                            seatbelt_mask = np.zeros([192, 192, 4])
                            if random.random() < 0.5:
                                seatbelt_mask = cv2.line(seatbelt_mask, [0, 0], [semi_x, semi_y], 
                                    (1, 1, 1, 1), thickness, lineType=cv2.LINE_AA) 
                            else:
                                seatbelt_mask = cv2.line(seatbelt_mask, [random.randint(20, 45), random.randint(30, 70)], [semi_x, semi_y], 
                                    (1, 1, 1, 1), thickness, lineType=cv2.LINE_AA) 

                            seatbelt_mask = cv2.line(seatbelt_mask, [semi_x, semi_y], [end_x, end_y], 
                                    (1, 1, 1, 1) , thickness, lineType=cv2.LINE_AA)
                            seatbelt_mask = random_wave(seatbelt_mask)
                            seatbelt_mask = seatbelt_mask.astype(bool)

                            seatbelt_img[:, :, 0] = color_element*b*3
                            seatbelt_img[:, :, 1] = color_element*g*3
                            seatbelt_img[:, :, 2] = color_element*r*3
                            seatbelt_img[:, :, 3] = alpha_element
                            seatbelt_img[: ,:, :3] = gaussian_illumination(seatbelt_img[: ,:, :3])

                            seatbelt_img = np.where(seatbelt_mask, seatbelt_img, [0, 0, 0, 0])
                        

                        #obstacle
                        if random.random() < hyp.get('obstacle', 0):
                            obstacle_mask = np.zeros([seatbelt_img.shape[0], seatbelt_img.shape[1]])
                            mosaic_patch_size = (img.shape[1]*img.shape[0])**0.5
                            obstacle_thickness = int((mosaic_patch_size/8) + random.random()*(mosaic_patch_size/8))
                            obstacle_mask = cv2.line(obstacle_mask, [random.randint(int(seatbelt_img.shape[1]/2), seatbelt_img.shape[1]), random.randint(0, int(seatbelt_img.shape[0]/2))], 
                                    [random.randint(0, int(seatbelt_img.shape[1]/2)), random.randint(int(seatbelt_img.shape[0]/2), seatbelt_img.shape[0])], 
                                    (1), obstacle_thickness, lineType=cv2.LINE_AA) 
                            seatbelt_img[obstacle_mask==1, :] = 0

                        face_label = labels[0]
                        if int(face_label[4]) < img.shape[0]*0.8:
                            random_x_transpose = (random.random()-0.5)*(face_label[3]-face_label[1])
                            seat_x1_range = min(max(face_label[1]-(face_label[3]-face_label[1])+random_x_transpose, 0), img.shape[1])
                            seat_y1_range = min(max(face_label[4]-(face_label[4]-face_label[2])*random.random()*0.2, 0), img.shape[0])
                            seat_x2_range = min(max(face_label[3]+(face_label[3]-face_label[1])*(1.1)+random_x_transpose, 0), img.shape[1])
                            seat_y2_range = min(max(face_label[4]+(face_label[4]-face_label[2])*(1+random.random()), 0), img.shape[0])

                            if seatbelt_filename is not None and (seatbelt_filename.startswith('03') or seatbelt_filename.startswith('04')):
                                seat_x1_start = int(min(max(face_label[1]-(face_label[3]-face_label[1])*1.5+random_x_transpose, 0), img.shape[1]))
                                seat_y1_start = int(seat_y1_range)
                                seat_x2_start = int(min(max(face_label[3]+(face_label[3]-face_label[1])*1.5+random_x_transpose, 0), img.shape[1]))
                                seat_y2_start = int(seat_y2_range)
                            else:
                                seat_x1_start = int(min(max(face_label[1]-(face_label[3]-face_label[1])*(0.25+random.random()*0.5)+random_x_transpose, 0), img.shape[1]))
                                seat_y1_start = int(min(max(random.randint(int(face_label[2]), int(seat_y1_range)), 0), img.shape[0]))
                                seat_x2_start = int(min(max(face_label[3]+(face_label[3]-face_label[1])*(-0.25+random.random())+random_x_transpose, 0), img.shape[1]))
                                seat_y2_start = int(seat_y2_range)

                            color_element = 32+int(random.random()*128)
                            mosaic_patch_size = (img.shape[1]*img.shape[0])**0.5
                            thickness = int((mosaic_patch_size/16) + random.random()*(mosaic_patch_size/16))

                            x1 = int(seat_x1_range)
                            y1 = int(seat_y1_range)
                            x2 = min(max(int(seat_x1_range+(seat_x2_range-seat_x1_range)*(end_x/seatbelt_img.shape[1])), 0), img.shape[1])      
                            y2 = min(max(int(seat_y1_range+(seat_y2_range-seat_y1_range)*(end_y/seatbelt_img.shape[0])), 0), img.shape[0])     
                            
                            if (seat_x2_start-seat_x1_start) > 0 and (seat_y2_start-seat_y1_start) > 0 :
                                seatbelt_img = cv2.resize(seatbelt_img, ((seat_x2_start-seat_x1_start), (seat_y2_start-seat_y1_start)), interpolation=cv2.INTER_LINEAR)

                                try:
                                    for idx_x in range(seatbelt_img.shape[1]) :
                                        for idx_y in range(seatbelt_img.shape[0]) :

                                            color_sum = np.sum(seatbelt_img[idx_y][idx_x][0:3])

                                            if color_sum > 10 :
                                                seatbelt_img[idx_y][idx_x][0] = min(int(color_sum * b),255)
                                                seatbelt_img[idx_y][idx_x][1] = min(int(color_sum * g),255)
                                                seatbelt_img[idx_y][idx_x][2] = min(int(color_sum * r),255)
                                except:
                                    seatbelt_img = seatbelt_img

                                if random.randint(0,1) == 0 :
                                    seatbelt_img = cv2.flip(seatbelt_img, 1)
                                    seatbet_center_x = (seat_x1_start+seat_x2_start)/2
                                    x1 = seatbet_center_x + (seatbet_center_x - x1)
                                    x2 = seatbet_center_x + (seatbet_center_x - x2)

                                img_crop = img[seat_y1_start:seat_y2_start, seat_x1_start:seat_x2_start]
                                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2RGBA)

                                # Pillow 에서 Alpha Blending
                                seatbelt_img_pillow = Image.fromarray(seatbelt_img.astype(np.uint8))
                                img_crop_pillow = Image.fromarray(img_crop)
                                blended_pillow = Image.alpha_composite(img_crop_pillow, seatbelt_img_pillow)
                                blended_img=np.array(blended_pillow)  

                                # 원본 이미지에 다시 합치기
                                blended_img = cv2.cvtColor(blended_img, cv2.COLOR_RGBA2RGB)
                                img[seat_y1_start:seat_y2_start, seat_x1_start:seat_x2_start] = blended_img

                                labels = np.append(labels, np.array([[0, x1, y1, x2, y2]]), axis=0)  
                                segments.append(np.array([
                                    [x1, y1], 
                                    [x2, y1],  
                                    [x2, y2],
                                    [x1, y2]
                                    ]))

        labels9.append(labels)
        segments9.extend(segments)
        if self.pose_data is not None:
            poses9.extend(poses)

        # Image
        img9[y1a:y2a, x1a:x2a] = img[y1a - pady:, x1a - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = [int(random.uniform(0, ys)), int(random.uniform(0, xs))]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * ys, xc:xc + 2 * xs]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * max(xs, ys), out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    poses9 = [x - c if x.any() is not None else x for x in poses9]
    poses9 = np.array(poses9)
    if self.pose_data is not None:
        for x in poses9:
            if x.any() is not None:
                np.clip(x, 0, 2 * max(xs, ys), out=x)

    # Augment
    #img9, labels9, segments9 = remove_background(img9, labels9, segments9)
    img9, labels9, segments9 = copy_paste(img9, labels9, segments9, probability=self.hyp['copy_paste'])
    img9, labels9, segments9, poses9 = random_perspective(img9, labels9, segments9, poses9,
                                        degrees=self.hyp['degrees'],
                                        translate=self.hyp['translate'],
                                        scale=self.hyp['scale'],
                                        shear=self.hyp['shear'],
                                        perspective=self.hyp['perspective'],
                                        border=self.mosaic_border)  # border to remove


    return img9, labels9, poses9, segments9

def load_samples(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    if isinstance(self.img_size, tuple):
        ys = self.img_size[0]
        xs = self.img_size[1]
    else:
        ys = self.img_size
        xs = self.img_size
    yc, xc = [int(random.uniform(-self.mosaic_border[0], 2 * ys + self.mosaic_border[0])), \
            int(random.uniform(-self.mosaic_border[1], 2 * xs + self.mosaic_border[1]))]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((ys * 2, xs * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, xs * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(ys * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, xs * 2), min(ys * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * max(xs, ys), out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    #img4, labels4, segments4 = remove_background(img4, labels4, segments4)
    sample_labels, sample_images, sample_masks = sample_segments(img4, labels4, segments4, probability=0.5)

    return sample_labels, sample_images, sample_masks


def copy_paste(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        im_new = np.zeros(img.shape, np.uint8)
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        img[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img, labels, segments


def remove_background(img, labels, segments):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    h, w, c = img.shape  # height, width, channels
    im_new = np.zeros(img.shape, np.uint8)
    img_new = np.ones(img.shape, np.uint8) * 114
    for j in range(n):
        cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        
        i = result > 0  # pixels to replace
        img_new[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img_new, labels, segments


def sample_segments(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    sample_labels = []
    sample_images = []
    sample_masks = []
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = l[1].astype(int).clip(0,w-1), l[2].astype(int).clip(0,h-1), l[3].astype(int).clip(0,w-1), l[4].astype(int).clip(0,h-1) 
            
            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue
            
            sample_labels.append(l[0])
            
            mask = np.zeros(img.shape, np.uint8)
            
            cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
            sample_masks.append(mask[box[1]:box[3],box[0]:box[2],:])
            
            result = cv2.bitwise_and(src1=img, src2=mask)
            i = result > 0  # pixels to replace
            mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
            sample_images.append(mask[box[1]:box[3],box[0]:box[2],:])

    return sample_labels, sample_images, sample_masks


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), segments=(), poses=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    min_label_size = 64

    targets[:, 3] - targets[:, 1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    if isinstance(scale, float):
        s = random.uniform(1 - scale, 1.1 + scale)
    else:
        s = random.uniform(1 + scale[0], 1.1 + scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        #use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4)) 
        new_seg = []

        #if use_segments:  # warp segments
        segments = resample_segments(segments)  # upsample
        for i, segment in enumerate(segments):
            #segmentation transform
            xy_seg = np.ones((len(segment), 3))
            xy_seg[:, :2] = segment
            xy_seg = xy_seg @ M.T  # transform
            xy_seg = xy_seg[:, :2] / xy_seg[:, 2:3] if perspective else xy_seg[:, :2]  # perspective rescale or affine
            # clip
            #new[i] = segment2box(xy, width, height)
            new_seg.append(xy_seg)

            #box transform
            #else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
            
        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.1)# 0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

        segments = []
        for i_idx, i_bool in enumerate(i):
            if i_bool:
                segments.append(new_seg[i_idx])

    ##poses##
    pose_n = len(poses)
    if pose_n:
        xy = np.ones((pose_n * 17, 3))
        xy[:, :2] = poses.reshape(pose_n * 17, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        new = xy[:, :2].reshape(pose_n, 17, 2)
        # clip
        new_ = poses.copy()
        for pose_i in range(pose_n):
            if poses[pose_i].any() is not None:
                new_[pose_i][poses[pose_i]>0] = new[pose_i][poses[pose_i]>0]
                new_[pose_i][new_[pose_i, :, 0]<=0 , :] = -1
                new_[pose_i][new_[pose_i, :, 1]<=0 , :] = -1
                new_[pose_i][new_[pose_i, :, 0]>width , :] = -1
                new_[pose_i][new_[pose_i, :, 1]>height , :] = -1                
            else:
                new_[pose_i] = poses[pose_i].copy()
            new_[new_==-1] = np.nan

        # filter candidates
        new_poses = new_#[i]
        #i = valid_pose_box(targets, new_poses, min_points=3, upper_only=True)
        #targets = targets[i]
        poses = new_poses
    
    return img, targets, segments, poses

def valid_pose_box(box, pose, min_points=1, upper_only=False):
    valid_idx = []
    for i, p in enumerate(pose):
        is_valid = np.ones(17)
        is_valid[[0,1,2]] = 5#eyes & nose
        is_valid[[3,4,5,6]] = 2#ears & shoulders
        is_valid[[7,8]] = 1.5#elbows
        if upper_only:    
              is_valid[[13,14,15,16]] = 0#legs
        else:
              is_valid[[11,12]] = 2#hips

        is_valid[p[:, 0] < box[i, 1]] = 0
        is_valid[p[:, 0] > box[i, 3]] = 0
        is_valid[p[:, 1] < box[i, 2]] = 0
        is_valid[p[:, 1] > box[i, 4]] = 0
        if np.sum(is_valid) >= min_points:
            valid_idx.append(i)

    return valid_idx


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area
    

def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels
    

def pastein(image, labels, sample_labels, sample_images, sample_masks):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
    for s in scales:
        if random.random() < 0.2:
            continue
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)   
        
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        if len(labels):
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area     
        else:
            ioa = np.zeros(1)
        
        if (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin+20) and (ymax > ymin+20):  # allow 30% obscuration of existing labels
            sel_ind = random.randint(0, len(sample_labels)-1)
            hs, ws, cs = sample_images[sel_ind].shape
            r_scale = min((ymax-ymin)/hs, (xmax-xmin)/ws)
            r_w = int(ws*r_scale)
            r_h = int(hs*r_scale)
            
            if (r_w > 10) and (r_h > 10):
                r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                temp_crop = image[ymin:ymin+r_h, xmin:xmin+r_w]
                m_ind = r_mask > 0
                if m_ind.astype(np.int).sum() > 60:
                    temp_crop[m_ind] = r_image[m_ind]
                    box = np.array([xmin, ymin, xmin+r_w, ymin+r_h], dtype=np.float32)
                    if len(labels):
                        labels = np.concatenate((labels, [[sample_labels[sel_ind], *box]]), 0)
                    else:
                        labels = np.array([[sample_labels[sel_ind], *box]])
                              
                    image[ymin:ymin+r_h, xmin:xmin+r_w] = temp_crop

    return image, labels


def dark_pastein(image, labels, sample_images, sample_masks):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
    for s in scales:
        if random.random() < 0.2:
            continue
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)   
        
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        if len(labels):
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area     
        else:
            ioa = np.zeros(1)
        
        if (ioa < 0.30).all() and len(sample_images) and (xmax > xmin+20) and (ymax > ymin+20):  # allow 30% obscuration of existing labels
            sel_ind = random.randint(0, len(sample_images)-1)
            hs, ws, cs = sample_images[sel_ind].shape
            r_scale = min((ymax-ymin)/hs, (xmax-xmin)/ws)
            r_w = int(ws*r_scale)
            r_h = int(hs*r_scale)
            
            if (r_w > 10) and (r_h > 10):
                r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                r_image = adjust_gamma(r_image, 0.05)
                temp_crop = image[ymin:ymin+r_h, xmin:xmin+r_w]
                m_ind = r_mask > 0
                if m_ind.astype(np.int).sum() > 60:
                    temp_crop[m_ind] = r_image[m_ind]                              
                    image[ymin:ymin+r_h, xmin:xmin+r_w] = temp_crop

    return labels

class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        import albumentations as A

        self.transform = A.Compose([
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.01),
            A.RandomGamma(gamma_limit=[80, 120], p=0.01),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.ImageCompression(quality_lower=75, p=0.01),],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

            #logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)

'''
def extract_boxes(path='../coco/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # Convert detection dataset into classification dataset, with one directory per class

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'

def autosplit(path='../coco', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit('../coco')
    Arguments
        path:           Path to images directory
        weights:        Train, val, test weights (list)
        annotated_only: Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in img_formats], [])  # image files only
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file    
'''
    
def load_segmentations(self, index):
    key = '/work/handsomejw66/coco17/' + self.img_files[index]
    #print(key)
    # /work/handsomejw66/coco17/
    return self.segs[key]

###########################mask/segmentation################################

def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygon2line(img_size, polygons, color=1, downsample_ratio=1, thickness=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    isClosed = True
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.polylines(mask, polygons, isClosed, color=color, thickness=thickness)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2lines(img_size, polygons, color, downsample_ratio=1, thickness=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2line(img_size, [polygons[si].reshape(-1)], color, downsample_ratio, thickness)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(img_size, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros((img_size[0] // downsample_ratio, img_size[1] // downsample_ratio),
            dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(
            img_size,
            [segments[si].reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index