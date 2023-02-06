target="/data/NCAP_videos_drives"
echo ${target##*/}
python3 detect.py --weights weights/smoke.pt --conf 0.5 --iou-thres 0.1 --source ${target} --name ${target##*/} --save-json --save-txt --img-size 608 800 --device 1 --frame-ratio 30 --save-frame
