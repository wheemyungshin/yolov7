target="/data/PITTA_2ND_ir_smoke_frames"
numbers=("0118_night_01" "0118_night_02" "0118_night_03" "0118_night_04" "0119_night_01" "0119_night_02" "0119_night_03" "0119_night_04" "0119_night_05" "0120_night_01" "0120_night_02" "0120_night_03" "0120_night_04" "0126_night_01" "0126_night_02" "0126_night_03" "0126_night_04" "0126_night_05" "0126_night_06")
subjects=("cf_best" "cf_good" "cf_bad" "rv_best" "rv_good" "rv_bad" "ws_best" "ws_good" "ws_bad" "db_bad" "rvm_best" "rvm_good" "rvm_bad" "wh_best" "wh_good" "wh_bad")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do
                if [ -d "${target}/${number}/${subject}" ]; then
                        python3 detect.py --weights weights/yolov7-e6e_memryx-update_s640.pt --conf 0.4 --iou-thres 0.65 --source ${target}/${number}/${subject} --name BF_${target##*/}_${number##*/}_${subject##*/} --save-json --save-txt --img-size 576 960 --device 4 --agnostic-nms --frame-ratio 30 --save-frame
                fi
        done
done