source ~/.bashrc

target="weights_smoke_n78_manual_testtrain_yolov7-tiny-liter_nomp"
for w in $target/*; do
        if [[ "$w" == *epoch*.pt ]]; then
            weight=${w##*/}
            echo "${weight%.*}"

            basename="${weight%.*}"

            python export.py --weights ${target}/${basename}.pt --simplify --iou-thres 0.2 --conf-thres 0.01 --img-size 128 128
            python onnx_head_remover.py --type yolov7-tiny-liter_nomp --model-name ${basename}_no_opt_128_128.onnx --dir ${target}


        fi
done

