source ~/.bashrc

target="weights_smoke_n78_manual_testtrain_yolov7-mobilenetv4"
for w in $target/*; do
        if [[ "$w" == *epoch*.pt ]]; then
            weight=${w##*/}
            echo "${weight%.*}"

            basename="${weight%.*}"

            python export.py --weights ${target}/${basename}.pt --simplify --iou-thres 0.2 --conf-thres 0.01 --img-size 128 128
            python onnx_head_remover.py --type mobilenetv4 --model-name ${basename}_no_opt_128_128.onnx --dir ${target}

        fi
done

