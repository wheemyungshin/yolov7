target="/data/valid_ciga_data_clean"
numbers=("NCAP_videos" "NCAP_videos_drives" "NCAP_videos_else" "NCAP_videos_models")
for number in ${numbers[@]}; do
        echo ${target##*/}_${number##*/}
        python3 detect.py --weights weights/yolov7-e6e_memryx-update_s640.pt --conf 0.5 --iou-thres 0.1 --source ${target}/${number} --name ${target##*/}_${number##*/} --save-json --save-txt --img-size 576 768 --device 5 --frame-ratio 30 --save-frame
done