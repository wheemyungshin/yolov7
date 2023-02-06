target="/data/PITTA_2nd_0126_divided"
numbers=("0126_day_02" "0126_day_03" "0126_day_04" "0126_day_05" "0126_day_06" "0126_night_01" "0126_night_02" "0126_night_03" "0126_night_04" "0126_night_05" "0126_night_06")
subjects=("db_bad" "rvm_best" "rvm_good" "rvm_bad" "wh_best" "wh_good" "wh_bad")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do
                echo ${target##*/}_${number##*/}_${subject}
                python3 detect.py --weights weights/yolov7-e6e_memryx-update_s640.pt --conf 0.2 --source ${target}/${number}/${subject} --name ${target##*/}_${number##*/}_${subject} --save-json --save-txt --img-size 576 1024 --device 0,1,2,3 --agnostic-nms --frame-ratio 30
        done
done