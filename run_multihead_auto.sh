target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
                python3 detect_multihead.py --weights runs/train/test_multihead_yolov7-tiny_smoke_all_BFC_s1924/weights/epoch_039.pt --conf 0.4 --source $dir --name test_multihead/${dir##*/} --img-size 192 320 --device 0 --agnostic-nms --save-frame --head-num 3
        fi
done