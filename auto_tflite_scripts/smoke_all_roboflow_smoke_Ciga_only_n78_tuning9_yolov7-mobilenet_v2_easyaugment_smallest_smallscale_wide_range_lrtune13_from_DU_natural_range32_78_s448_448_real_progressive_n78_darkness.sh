source ~/.bashrc
conda activate yolov7

target="runs/train/smoke_all_roboflow_smoke_Ciga_only_n78_tuning9_yolov7-mobilenet_v2_easyaugment_smallest_smallscale_wide_range_lrtune13_from_DU_natural_range32_78_s448_448_real_progressive_n78_darkness/weights"
save_tag="modified_GH_"
for w in $target/*; do
        if [[ "$w" == *epoch*.pt ]]; then
            weight=${w##*/}
            echo "${weight%.*}"

            basename="${weight%.*}"

            python export.py --weights ${target}/${basename}.pt --simplify --iou-thres 0.2 --conf-thres 0.01 --img-size 192 192
            python onnx_head_remover.py --type mobilenet_v2_192 --model-name ${basename}_no_opt_192_192.onnx --dir ${target} --save-tag ${save_tag}

            conda deactivate
            conda activate onnx2tf
            onnx2tf -i ${target}/${save_tag}${basename}_no_opt_192_192.onnx -oiqt

            conda deactivate
            conda activate yolov7

            python tflite_runner2_map_check.py --data data/smoke_all_ciga_n78_testset_manual.yaml \
                --img 192 --batch 1 --conf 0.05 --iou 0.65 \
                --weights saved_model/${save_tag}${basename}_no_opt_192_192_integer_quant.tflite \
                --weights2 ../onnx2tf/saved_model/NMS_mobilenet_v2_s192_192_float32.tflite \
                --result-txt ${target}
                
        fi
done

