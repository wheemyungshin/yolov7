target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
                python3 detect.py --weights runs/train/0127memryx_mafa-yolov7-tiny_body_only_s192_3206/weights/epoch_039.pt --conf 0.4 --source $dir --name ${dir##*/} --save-txt --img-size 192 320 --device 0 --agnostic-nms --save-frame
        fi
done