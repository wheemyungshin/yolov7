source ~/.bashrc
conda activate yolov7

target="weights_phone_candis"
for w in $target/*; do
        if [[ "$w" == *_integer_quant.tflite* ]]; then
            echo "${w}"

            python tflite_runner2_map_check.py --data data/smoke_all_phone_n78_testset_manual.yaml \
                --img 128 --batch 1 --conf 0.05 --iou 0.65 \
                --weights ${w} \
                --weights2 ../onnx2tf/saved_model/NMS_mobilenet_s128_128_float32.tflite \
                --result-txt ${target}
                
        fi
done

