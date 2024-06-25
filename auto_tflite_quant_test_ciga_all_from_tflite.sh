source ~/.bashrc
conda activate yolov7

target="weights_ciga_candis"
for w in $target/*; do
        if [[ "$w" == *_integer_quant.tflite* ]]; then
            echo "${w}"

            python tflite_runner2_map_check.py \
                --data data/smoke_all_ciga_n78_testset_manual.yaml \
                 --img 192 --batch 1 --conf 0.05 --iou 0.65 \
                --weights ${w} \
                --weights2 ../onnx2tf/saved_model/NMS_mobilenet_v2_s192_192_float32.tflite \
                --result-txt ${target}
                
        fi
done

