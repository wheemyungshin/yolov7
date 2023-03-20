target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
                python3 detect.py --weights weights/seatbelt_crop_yolov7-tiny_s384_384.pt --conf 0.4 --source $dir --name seatbelt_smokeall_bodycrop/${dir##*/} --img-size 384 384 --device 0 --agnostic-nms --save-frame
        fi
done