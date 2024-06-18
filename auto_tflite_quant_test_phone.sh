
source ~/.bashrc
conda activate yolov7

basename="GG_phone_mobilenet_e497"
echo weights/${basename}.pt

python export.py --weights weights/${basename}.pt --simplify --iou-thres 0.2 --conf-thres 0.01 --img-size 128 128
python onnx_head_remover.py --type mobilenet_128 --model-name ${basename}_no_opt_128_128.onnx

conda deactivate
conda activate onnx2tf
onnx2tf -i weights/modified_${basename}_no_opt_128_128.onnx -oiqt

conda deactivate
conda activate yolov7

python tflite_runner2_map_check.py --data data/smoke_all_P_val_n78.yaml --img 128 --batch 1 --conf 0.05 --iou 0.65 \
    --weights saved_model/modified_${basename}_no_opt_128_128_integer_quant.tflite \
    --weights2 ../onnx2tf/saved_model/NMS_mobilenet_s128_128_float32.tflite