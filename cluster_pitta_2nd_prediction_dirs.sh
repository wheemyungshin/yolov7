for i in {2..15}; do
    echo "$i"
done

target="/data/PITTA_2ND_divided"
numbers=("0118_day_01" "0118_day_02" "0118_day_03" "0118_night_01" "0118_night_02" "0118_night_03" "0118_night_04" "0119_day_01" "0119_day_02" "0119_day_03" "0119_day_04" "0119_night_01" "0119_night_02" "0119_night_03" "0119_night_04" "0119_night_05" "0120_day_01" "0120_day_02" "0120_day_03" "0120_day_04" "0120_night_01" "0120_night_02" "0120_night_03" "0120_night_04")
subjects=("cf_best" "cf_good" "cf_bad" "rv_best" "rv_good" "rv_bad" "ws_best" "ws_good" "ws_bad")
for number in ${numbers[@]}; do
        for subject in ${subjects[@]}; do        
                for i in {2..15}; do                
                        len=$(ls runs/detect/${target##*/}_${number##*/}_${subject##*/}$i | wc -l)
                        if [ $len -lt 3 ]; then
                                echo "$len"
                                rm -r runs/detect/${target##*/}_${number##*/}_${subject##*/}$i
                        fi
                done
        done
done