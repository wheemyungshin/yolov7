target="/data"
numbers=("roboflow_head")
subjects=("test" "train" "valid")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do
                if [ -d "${target}/${number}/${subject}" ]; then
                        python3 detect.py --weights weights/yolov7-e6e_memryx-update_s640.pt --conf 0.55 --iou-thres 0.65 --source ${target}/${number}/${subject}/images --name BF_${target##*/}_${number##*/}_${subject##*/} --img-size 640 640 --device 0 --agnostic-nms --save-txt --save-frame
                fi
        done
done