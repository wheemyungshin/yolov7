target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
                python3 detect.py --weights weights/0127memryx_mafa-yolov7-tiny_s192_320_bf.pt  --conf 0.4 --source $dir --name 0127memryx_mafa-yolov7-tiny_s192_320_BF/${dir##*/} --img-size 192 320 --device 0 --agnostic-nms --save-frame

        fi
done