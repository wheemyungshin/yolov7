target="/data/PITTA_2nd_0126_divided"
numbers=("0126_day_01" "0126_day_02" "0126_day_03" "0126_day_04" "0126_day_05" "0126_day_06")
subjects=("cf_best" "cf_good" "cf_bad" "db_bad" "rvm_best" "rvm_good" "rvm_bad" "wh_best" "wh_good" "wh_bad")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do
                if [ -d "${target}/${number}/${subject}" ]; then
                        python3 detect.py --weights weights/smoke_all_yolov7-tiny_s384_640_bc.pt --conf 0.5 --iou-thres 0.3 --source ${target}/${number}/${subject} --name PITTA_2nd_0126_divided_0126_day_C/${target##*/}_${number##*/}_${subject##*/} --save-txt --img-size 384 640 --device 0,1,2,3 --agnostic-nms --frame-ratio 30 --save-frame --classes 1
                fi
        done
done