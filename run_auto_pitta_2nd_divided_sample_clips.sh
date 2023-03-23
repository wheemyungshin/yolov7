target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
                python3 detect.py --weights weights/smoke_all_crop_yolov7_easyaug_s576_576_c.pt --conf 0.4 --source $dir --name smoke_all_crop_yolov7_easyaug_s576_576_c/${dir##*/} --img-size 576 1024 --device 0 --agnostic-nms --save-frame
        fi
done