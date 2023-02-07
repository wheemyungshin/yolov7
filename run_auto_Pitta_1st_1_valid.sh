target="/data/valid_ciga_data_clean"
numbers=("PERSON_N_211006_Truck" "PERSON_N_211006_dwkim_day" "PERSON_N_211006_dwkim_night" "PERSON_N_211006_stkim_day" "PERSON_N_211006_stkim_night" "PERSON_N_211007_Truck" "PERSON_N_211007_dwkim_day" "PERSON_N_211007_dwkim_night" "PERSON_N_211007_stkim_day" "PERSON_N_211007_stkim_night" "PERSON_N_211008_dwkim_day" "PERSON_N_211008_dwkim_night" "PERSON_N_211008_stkim_day" "PERSON_N_211008_stkim_night")
for number in ${numbers[@]}; do
        echo ${target##*/}_${number##*/}
        python3 detect.py --weights weights/yolov7-e6e_memryx-update_s640.pt --conf 0.5 --iou-thres 0.1 --source ${target}/${number} --name ${target##*/}_${number##*/} --save-json --save-txt --img-size 576 1024 --device 0 --frame-ratio 30 --save-frame
done