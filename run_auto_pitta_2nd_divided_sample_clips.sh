target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
<<<<<<< HEAD
                python3 detect.py --weights weights/smoke_all_crop_yolov7-tiny_s576_576_c.pt  --conf 0.4 --source $dir --name smoke_all_crop_yolov7-tiny_s576_576/${dir##*/} --img-size 576 960 --device 0 --agnostic-nms --save-frame
=======
                python3 detect.py --weights weights/smoke_all_crop_yolov7_s192_192.pt  --conf 0.4 --source $dir --name smoke_all_crop_yolov7_s192_192/${dir##*/} --img-size 192 320 --device 0 --agnostic-nms --save-frame

>>>>>>> 7a9da1a7629318414acd052ee5e28405ceac7816
        fi
done