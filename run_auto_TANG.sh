target="/data/TANG_smoke"
numbers=("01" "02" "03")
for number in ${numbers[@]}; do
        echo ${target##*/}_${number##*/}
        python3 detect.py --weights weights/smoke.pt --conf 0.5 --iou-thres 0.1 --source ${target}/${number} --name ${target##*/}_${number##*/} --save-json --save-txt --img-size 480 640 --device 5 --frame-ratio 30 --save-frame
done