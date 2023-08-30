target="runs/train"
for supdir in $target/*; do
        if [ -d "$supdir" ]; then
                len=$(ls $supdir | wc -l)
                if [ $len -lt 5 ]; then
                        echo $supdir
                        echo $(ls $supdir)
                        echo $len
                        rm -r $supdir
                fi
        fi
done