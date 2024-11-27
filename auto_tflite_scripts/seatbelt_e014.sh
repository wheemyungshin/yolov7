source ~/.bashrc
conda activate yolov7

target="seatbelt1125_bottomcrop_e004_qat_4th"
for w in $target/*; do
        if [[ "$w" == *epoch*.pt ]]; then
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

            python tflite_runner2_map_check.py --data data/auto_seatbelt_1121_qa_true.yaml \
                --img 96 --batch 1 --conf 0.6 --iou 0.65 \
                --weights saved_model/modified_${basename}_no_opt_96_96_float16.tflite \
                --weights2 saved_model/NMS_mobilenet_t_96_96_float32.tflite \
                --result-txt ${target} --merge-label 0                
        fi
done

