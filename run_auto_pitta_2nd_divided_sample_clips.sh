target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
                python3 detect.py --weights weights/0127memryx_mafa-yolov7-tiny_face_only_rot_s192_320.pt  --conf 0.4 --source $dir --name 0127memryx_mafa-yolov7-tiny_face_only_rot_s192_320/${dir##*/} --img-size 192 320 --device 0 --agnostic-nms --save-frame

        fi
done