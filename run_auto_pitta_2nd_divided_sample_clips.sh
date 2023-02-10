target="/data/PITTA_2ND_divided_sample_clips"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
<<<<<<< HEAD
                python3 detect.py --weights weights/0127memryx_mafa-yolov7-tiny_face_only_rot_s192_320.pt  --conf 0.4 --source $dir --name 0127memryx_mafa-yolov7-tiny_face_only_rot_s192_320/${dir##*/} --img-size 192 320 --device 0 --agnostic-nms --save-frame
=======
                python3 detect.py --weights weights/tang_ncap_pitta1st_2nd_smoke_tune_yolov7-tiny_s192_320.pt  --conf 0.4 --source $dir --name tang_ncap_pitta1st_2nd_smoke_tune_yolov7-tiny_s192_320_to_s192_192/${dir##*/} --img-size 192 192 --device 0 --agnostic-nms --save-frame
>>>>>>> 1769e88ed1f68f25236b8cacd55b411b02eaf514
        fi
done