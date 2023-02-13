target="/data/DMC-200/PERSON_N"
numbers=("211007_dwkim_day" "211007_dwkim_night" "211007_stkim_day")
for number in ${numbers[@]}; do
        echo ${target##*/}_${number##*/}
        python3 detect.py --weights weights/smoke.pt --conf 0.5 --iou-thres 0.1 --source ${target}/${number} --name ${target##*/}_${number##*/} --save-json --save-txt --img-size 544 960 --device 2 --frame-ratio 30 --save-frame
done