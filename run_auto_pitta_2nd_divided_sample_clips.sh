target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
                python3 detect.py --weights weights/tang_ncap_pitta1st_2nd_tune_yolov7-tiny_s192_320.pt  --conf 0.4 --source $dir --name tang_ncap_pitta1st_2nd_tune_yolov7-tiny_s192_320/${dir##*/} --img-size 192 320 --device 0 --agnostic-nms --save-frame
        fi
done