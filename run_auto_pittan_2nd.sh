target="/data/PITTA_2ND_divided"
numbers=("0118_day_01" "0118_day_02" "0118_day_03" "0118_night_01" "0118_night_02" "0118_night_03" "0118_night_04" "0119_day_01" "0119_day_02" "0119_day_03" "0119_day_04" "0119_night_01" "0119_night_02" "0119_night_03" "0119_night_04" "0119_night_05" "0120_day_01" "0120_day_02" "0120_day_03" "0120_day_04" "0120_night_01" "0120_night_02" "0120_night_03" "0120_night_04")
subjects=("cf" "rv" "ws")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do
                echo ${target##*/}_${number##*/}_${subject}
                python3 detect.py --weights weights/yolov7-tiny_memryx-update_s192.pt --conf 0.2 --source ${target}/${number}/${subject} --name ${target##*/}_${number##*/}_${subject} --save-json --save-txt --img-size 192 256 --device 0,1,2,3 --agnostic-nms --frame-ratio 30
        done
done
