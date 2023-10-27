target="/data/roboflow_phone"
subjects=("test" "train" "valid")
for dir in $target/*; do
        for subject in ${subjects[@]}; do
                echo "BF_${dir##*/}/${subject##*/}" 
                if [ -d "${dir}/${subject}" ]; then
                        CUDA_VISIBLE_DEVICES=1 python3 detect.py --weights weights/yolov7-e6e_memryx-update_s640.pt --conf 0.55 --iou-thres 0.65 --source ${dir}/${subject}/images --name BF_${dir##*/}/${subject##*/} --img-size 640 640 --device 0 --agnostic-nms --save-txt --save-frame
                fi
        done
done