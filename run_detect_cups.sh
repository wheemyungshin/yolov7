target="/coco/cup_data/1_classification"
numbers=("paper_cups/paper_cups" "plastic_cups/image")
subjects=("MIX" "cap" "holder" "normal" "stick")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do
                echo ${target}
                echo ${target##*/}
                python3 detect.py --weights weights/yolov7_coco_cups_only_tune_s256.pt --conf 0.25 --source ${target}/${number}/${subject} --name ${target##*/}_${number##*/}_${subject} --save-json
        done
done