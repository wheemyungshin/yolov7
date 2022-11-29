target="/coco/cup_data/1_classification"
numbers=("paper_cups/paper_cups" "plastic_cups/image" "plastic_cups/video")
subjects=("MIX" "cap" "holder" "normal" "stick" "MIX_video")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do
                echo ${target##*/}_${number##*/}_${subject}
                python3 detect.py --weights weights/yolov7_coco_and_synthetic_cups_only_tune_s320.pt --conf 0.25 --source ${target}/${number}/${subject} --name ${target##*/}_${number##*/}_${subject} --save-json --save-txt
        done
done