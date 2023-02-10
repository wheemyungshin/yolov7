target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
                python3 detect_ciga_crop.py --body-weights weights/0127memryx_mafa-yolov7-tiny_body_only_s192_320_B.pt --weights weights/tang_ncap_pitta1st_2nd_smoke_tune_yolov7-tiny_s192_320.pt  --conf 0.4 --source $dir --name crop_body_tang_ncap_pitta1st_2nd_smoke_tune_yolov7-tiny_s192_320_to_s192_192/${dir##*/} --img-size 192 320 --device 0 --agnostic-nms --save-frame
        fi
done