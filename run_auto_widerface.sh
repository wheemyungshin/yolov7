target="/data/wider_face"
numbers=("WIDER_train" "WIDER_val")
subjects=("0--Parade" "12--Group" "16--Award_Ceremony" "2--Demonstration" "23--Shoppers" "27--Spa" "30--Surgeons" "34--Baseball" "38--Tennis" "41--Swimming" "45--Balloonist" "1--Handshaking" "13--Interview" "17--Ceremony" "20--Family_Group" "24--Soldier_Firing" "28--Sports_Fan" "31--Waiter_Waitress" "35--Basketball" "39--Ice_Skating" "42--Car_Racing" "46--Jockey" "10--People_Marching" "14--Traffic" "18--Concerts" "21--Festival" "25--Soldier_Patrol" "29--Students_Schoolkids" "32--Worker_Laborer" "36--Football" "4--Dancing" "43--Row_Boat" "11--Meeting" "15--Stock_Market" "19--Couple" "22--Picnic" "26--Soldier_Drilling" "3--Riot" "33--Running" "37--Soccer" "40--Gymnastics" "44--Aerobics")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do
                echo ${target##*/}_${number##*/}_${subject}
                python3 detect.py --weights weights/yolov7-e6e.pt --conf 0.2 --source ${target}/${number}/${subject} --name ${target##*/}_${number##*/}_${subject} --save-json --save-txt --img-size 1280 1280 --device 4,5,6,7 --classes 0
        done
done