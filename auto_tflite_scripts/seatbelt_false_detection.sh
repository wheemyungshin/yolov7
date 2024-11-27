source ~/.bashrc
conda activate yolov7

target="weights_seatbelt1127"
for w in $target/*; do
        if [[ "$w" == *leftright*.pt ]]; then
            weight=${w##*/}
            echo "${weight%.*}"

            basename="${weight%.*}"

            python export.py --nc 1 --cfg cfg/training/yolov7-mobilenet.yaml --weights ${target}/${basename}.pt --simplify --iou-thres 0.5 --conf-thres 0.01 --img-size 96 96 --qat
            python onnx_head_remover.py --type mobilenet_t_96_96_test --model-name ${basename}_no_opt_96_96.onnx --dir ${target}

            conda deactivate
            conda activate onnx2tf
            onnx2tf -i ${target}/modified_${basename}_no_opt_96_96.onnx -oiqt

            conda deactivate
            conda activate yolov7

            python tflite_runner2_det_count.py --data ../data/auto_seatbelt_pittaqasmoke_false/images\
                --img 96 --batch 1 --conf 0.5 --iou 0.65 \
                --weights saved_model/modified_${basename}_no_opt_96_96_float16.tflite \
                --weights2 saved_model/NMS_mobilenet_t_96_96_float32.tflite \
                --result-txt ${target} --merge-label 0                
        fi
done

