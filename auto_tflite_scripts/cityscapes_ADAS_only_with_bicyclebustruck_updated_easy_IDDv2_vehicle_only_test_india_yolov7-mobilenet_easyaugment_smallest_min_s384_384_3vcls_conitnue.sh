source ~/.bashrc
conda activate yolov7

target="weights_cityscapes_ADAS_only_with_bicyclebustruck_updated_easy_IDDv2_vehicle_only_test_india_yolov7-mobilenet_easyaugment_smallest_min_s384_384_3vcls_conitnue"
for w in $target/*; do
        if [[ "$w" == *epoch*.pt ]]; then
            weight=${w##*/}
            echo "${weight%.*}"

            basename="${weight%.*}"

            python export.py --weights ${target}/${basename}.pt --simplify --iou-thres 0.2 --conf-thres 0.01 --img-size 128 224
            python onnx_head_remover.py --type mobilenet_128_224 --model-name ${basename}_no_opt_128_224.onnx --dir ${target}

            conda deactivate
            conda activate onnx2tf
            onnx2tf -i ${target}/modified_${basename}_no_opt_128_224.onnx -oiqt

            conda deactivate
            conda activate yolov7

            python tflite_runner2_map_check.py --data data/smoke_all_ciga_n78_testset_manual.yaml \
                --img 128 224 --batch 1 --conf 0.05 --iou 0.65 \
                --weights saved_model/modified_${basename}_no_opt_128_224_integer_quant.tflite \
                --weights2 saved_model/NMS_mobilenet_3cls_s128_224_noqat_nocfg_float32.tflite  \
                --result-txt ${target}
                
        fi
done

