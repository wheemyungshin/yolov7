source ~/.bashrc
conda activate yolov7

target="weights_smoke_n78_manual_testtrain_yolov7-tiny-lite_half_nomp"
for w in $target/*; do
        if [[ "$w" == *epoch*.onnx ]]; then
            weight=${w##*/}
            echo "${weight%.*}"

            basename="${weight%.*}"
            
            conda deactivate
            conda activate onnx2tf

            onnx2tf -i ${target}/${basename}.onnx -oiqt

            conda deactivate
            conda activate yolov7

            python tflite_runner2_map_check.py --data data/smoke_all_smoke_n78_testset_manual.yaml \
                --img 128 --batch 1 --conf 0.05 --iou 0.65 \
                --weights saved_model/${basename}_integer_quant.tflite \
                --weights2 ../onnx2tf/saved_model/NMS_mobilenet_s128_128_float32.tflite \
                --result-txt ${target}
                
        fi
done