source ~/.bashrc
conda activate yolov7

target="weights_alif"
for w in $target/*; do
        weight=${w##*/}
        echo "${weight%.*}"

        basename="${weight%.*}"

        if [[ "$w" == *qat*.pt ]]; then
                python export.py --nc 1 --cfg cfg/training/yolov7-mobilenet_t.yaml --weights ${target}/${basename}.pt --simplify --iou-thres 0.1 --conf-thres 0.01 --img-size 128 128 --qat
        else
                python export.py --weights ${target}/${basename}.pt --simplify --iou-thres 0.1 --conf-thres 0.01 --img-size 128 128
        fi

        python onnx_head_remover.py --type mobilenet_t_128_128 --model-name ${basename}_no_opt_128_128.onnx --dir ${target}

        conda deactivate
        conda activate onnx2tf
        onnx2tf -i ${target}/modified_${basename}_no_opt_128_128.onnx -oiqt

        conda deactivate
        conda activate yolov7

        python tflite_runner2_map_check.py --data data/0910_head_body_updated_shut_aliftest.yaml \
        --img 128 --batch 1 --conf 0.05 --iou 0.65 \
        --weights saved_model/modified_${basename}_no_opt_128_128_integer_quant.tflite \
        --weights2 ../onnx2tf/saved_model/NMS_yolov7-tiny-liter_nomp_s128_128_float32.tflite \
        --result-txt ${target}
                
done

