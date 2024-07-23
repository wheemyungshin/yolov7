source ~/.bashrc
conda activate yolov7

target="weights_smoke_all_P_n78_tuning_only13_yolov7-tiny-liter_nomp_easyaugment_smallest_smallscale_wide_range_lrtune7_3_from_trainall_s256_256_n78_darkness"
save_tag="modified_GK_"
for w in $target/*; do
        if [[ "$w" == *epoch*.pt ]]; then
            weight=${w##*/}
            echo "${weight%.*}"

            basename="${weight%.*}"

            python export.py --weights ${target}/${basename}.pt --simplify --iou-thres 0.2 --conf-thres 0.01 --img-size 128 128
            python onnx_head_remover.py --type yolov7-tiny-liter_nomp --model-name ${basename}_no_opt_128_128.onnx --dir ${target} --save-tag ${save_tag}

            conda deactivate
            conda activate onnx2tf
            onnx2tf -i ${target}/${save_tag}${basename}_no_opt_128_128.onnx -oiqt

            conda deactivate
            conda activate yolov7

            python tflite_runner2_map_check.py --data data/smoke_all_phone_n78_testset_manual.yaml \
                --img 128 --batch 1 --conf 0.05 --iou 0.65 \
                --weights saved_model/${save_tag}${basename}_no_opt_128_128_integer_quant.tflite \
                --weights2 ../onnx2tf/saved_model/NMS_yolov7-tiny-liter_nomp_s128_128_float32.tflite \
                --result-txt ${target}
                
        fi
done
