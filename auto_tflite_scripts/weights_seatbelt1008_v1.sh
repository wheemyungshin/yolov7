source ~/.bashrc
conda activate yolov7

target="weights_seatbelt1008_v1"
for w in $target/*; do
        if [[ "$w" == *epoch*.pt ]]; then
            weight=${w##*/}
            echo "${weight%.*}"

            basename="${weight%.*}"

            python export.py --weights ${target}/${basename}.pt --simplify --iou-thres 0.2 --conf-thres 0.01 --img-size 128 128
            python onnx_head_remover.py --type mobilenet_t_128_128_test --model-name ${basename}_no_opt_128_128.onnx --dir ${target}

            conda deactivate
            conda activate onnx2tf
            onnx2tf -i ${target}/modified_${basename}_no_opt_128_128.onnx -oiqt

            conda deactivate
            conda activate yolov7                
        fi
done

