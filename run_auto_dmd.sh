target="/data/dmd/1"

targets=("/data/dmd/1" "/data/dmd/2" "/data/dmd/3" "/data/dmd/4" "/data/dmd/5")
for target in ${targets[@]}; do
        for supdir in $target/*; do
                if [ -d "$supdir" ]; then
                        for dir in $supdir/*; do
                                if [ -d "$dir" ]; then
                                echo $dir
                                echo ${dir##*/}
                                python3 detect.py --weights weights/yolov7-e6e_memryx-update_s640.pt --conf 0.55 --source $dir --name dmd_auto_label_BF/${dir##*/} --img-size 640 640 --device 0 --agnostic-nms --save-frame --save-txt
                                fi
                        done
                fi
        done
done