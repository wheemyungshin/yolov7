target="/data/DMC-200/PERSON_N"
numbers=("211008_stkim_day" "211008_stkim_night")
for number in ${numbers[@]}; do
        echo ${target##*/}_${number##*/}
        python3 detect.py --weights weights/smoke.pt --conf 0.5 --iou-thres 0.1 --source ${target}/${number} --name ${target##*/}_${number##*/} --save-json --save-txt --img-size 544 960 --device 3 --frame-ratio 30 --save-frame
done