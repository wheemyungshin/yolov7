target="/data/PITTA_2ND_divided"
numbers=("0118_day_01" "0118_day_02" "0118_day_03" "0118_night_01" "0118_night_02" "0118_night_03" "0118_night_04" "0119_day_01" "0119_day_02" "0119_day_03" "0119_day_04" "0119_night_01" "0119_night_02" "0119_night_03" "0119_night_04" "0119_night_05" "0120_day_01" "0120_day_02" "0120_day_03" "0120_day_04" "0120_night_01" "0120_night_02" "0120_night_03" "0120_night_04")
subjects=("cf_best" "cf_good" "cf_bad" "rv_best" "rv_good" "rv_bad" "ws_best" "ws_good" "ws_bad")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do
                counter=15
                video_files=${target}/${number}/${subject}/*
                len=(${target}/${number}/${subject}/*)
                len=${#len[@]}
                echo $video_files
                for video_file in $video_files; do
                        echo $((counter++))
                        echo $len
                        if [ $counter -gt $len ]; then
                                echo ${target##*/}_${number##*/}_${subject##*/}_${video_file##*/}
                                python3 detect.py --weights weights/0127memryx_mafa-yolov7-tiny.pt --conf 0.2 --source ${video_file} --name ${target##*/}_${number##*/}_${subject##*/} --save-json --save-txt --img-size 192 352 --device 0,1,2,3 --agnostic-nms --frame-ratio 30
                        fi
                done
        done
done
