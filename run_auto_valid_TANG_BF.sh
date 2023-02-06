target="/data/valid_ciga_data_clean"
numbers=("TANG_smoke_01" "TANG_smoke_02" "TANG_smoke_03")
for number in ${numbers[@]}; do
        echo ${target##*/}_${number##*/}
        python3 detect.py --weights weights/yolov7-e6e_memryx-update_s640.pt --conf 0.5 --iou-thres 0.1 --source ${target}/${number} --name ${target##*/}_${number##*/} --save-json --save-txt --img-size 512 640 --device 5 --frame-ratio 30 --save-frame
done