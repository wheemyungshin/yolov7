target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
                python3 detect.py --weights runs/train/smoke_all_crop_tune_yolov7-tiny_boxmargin1_8_s192_1922/weights/epoch_109.pt  --conf 0.4 --source $dir --name smoke_all_crop_tune_yolov7-tiny_boxmargin1_8_s192_192_epoch_109/${dir##*/} --img-size 192 320 --device 0 --agnostic-nms --save-frame


        fi
done