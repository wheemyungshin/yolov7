target="/data/PITTA_2ND_0126_ir_smoke_frames"

for sub_dir in $target/*; do
        for dir in $sub_dir/*; do
                if [ -d "$dir" ]; then
                        echo $dir
                        echo auto_ciga_PITTA_2ND_0126_ir_smoke_frames/${sub_dir##*/}_${dir##*/}
                        python3 detect.py --weights weights/smoke_all_crop_yolov7_s192_192.pt --conf 0.3 --source $dir --name auto_ciga_PITTA_2ND_0126_ir_smoke_frames/${sub_dir##*/}_${dir##*/} --img-size 192 320 --device 0 --agnostic-nms --save-frame --save-txt
                fi
        done
done