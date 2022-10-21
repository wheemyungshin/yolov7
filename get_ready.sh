docker run --name yolov7 -it -v /ssd2/wmshin/data/coco/:/coco -v /ssd2/wmshin/yolov7/:/yolov7 --shm-size=64g --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all nvcr.io/nvidia/pytorch:21.08-py3

###docker attach yolov7###
apt update
apt install -y zip htop screen libgl1-mesa-glx

pip install seaborn thop

cd /yolov7

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==4.5.5.64
