target="/mnt/hdd1/DB/PITTA_2ND_divided"

for subpath in $target/0126_night_*; do
        for dir in $subpath/*; do
                if [ -d "$dir" ]; then
                        echo $dir
                        echo ${subpath##*/}_${dir##*/}
                        python3 detect.py --weights weights/tang_ncap_pitta1st_2nd-yolov7-tiny_tune_s192_320.pt --conf 0.5 --source $dir --name ${subpath##*/}_${dir##*/} --save-json --save-txt --img-size 192 352 --device 0 --agnostic-nms --frame-ratio 30 --save-frame
                fi
        done
done
for subpath in $target/0120_*; do
        for dir in $subpath/*; do
                if [ -d "$dir" ]; then
                        echo $dir
                        echo ${subpath##*/}_${dir##*/}
                        python3 detect.py --weights weights/tang_ncap_pitta1st_2nd-yolov7-tiny_tune_s192_320.pt --conf 0.5 --source $dir --name ${subpath##*/}_${dir##*/} --save-json --save-txt --img-size 192 352 --device 0 --agnostic-nms --frame-ratio 30 --save-frame
                fi
        done
done
for subpath in $target/0119_*; do
        for dir in $subpath/*; do
                if [ -d "$dir" ]; then
                        echo $dir
                        echo ${subpath##*/}_${dir##*/}
                        python3 detect.py --weights weights/tang_ncap_pitta1st_2nd-yolov7-tiny_tune_s192_320.pt --conf 0.5 --source $dir --name ${subpath##*/}_${dir##*/} --save-json --save-txt --img-size 192 352 --device 0 --agnostic-nms --frame-ratio 30 --save-frame
                fi
        done
done
for subpath in $target/0118_*; do
        for dir in $subpath/*; do
                if [ -d "$dir" ]; then
                        echo $dir
                        echo ${subpath##*/}_${dir##*/}
                        python3 detect.py --weights weights/tang_ncap_pitta1st_2nd-yolov7-tiny_tune_s192_320.pt --conf 0.5 --source $dir --name ${subpath##*/}_${dir##*/} --save-json --save-txt --img-size 192 352 --device 0 --agnostic-nms --frame-ratio 30 --save-frame
                fi
        done
done