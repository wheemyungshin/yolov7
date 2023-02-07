target="/data/smoke_data"
numbers=("images")
subjects=("00" "01")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do
                if [ -d "${target}/${number}/${subject}" ]; then
                        python3 detect.py --weights weights/yolov7-e6e_memryx-update_s640.pt --conf 0.4 --iou-thres 0.65 --source ${target}/${number}/${subject} --name BF_${target##*/}_${number##*/}_${subject##*/} --save-json --save-txt --img-size 640 640 --device 0 --agnostic-nms --frame-ratio 30 --save-frame
                fi
        done
done