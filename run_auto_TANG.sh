target="/data/TANG_smoke"
numbers=("01" "02" "03")
for number in ${numbers[@]}; do
        echo ${target##*/}_${number##*/}
        #python3 detect.py --weights weights/yolov7-e6e_memryx-update_s640.pt --conf 0.5 --source ${target}/${number} --name TANG_body_auto/${target##*/}_${number##*/} --img-size 512 640 --device 0 --agnostic-nms --save-frame --save-txt
        python3 detect.py --weights weights/seatbelt_crop_yolov7-tiny_s384_384.pt --conf 0.5 --source ${target}/${number} --name TANG_seatbelt_auto/${target##*/}_${number##*/} --img-size 320 384 --device 0 --agnostic-nms --save-frame --save-txt
done