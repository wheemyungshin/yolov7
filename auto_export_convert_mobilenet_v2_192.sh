
source ~/.bashrc
conda activate yolov7

basenames=("FX_ciga_mobilenet_v2_e019" "FX_ciga_mobilenet_v2_e099" "FX_ciga_mobilenet_v2_e159" "FX_ciga_mobilenet_v2_e219" "FX_ciga_mobilenet_v2_e299" "FX_ciga_mobilenet_v2_e309")
for basename in ${basenames[@]}; do
    echo weights/${basename}.pt

    python export.py --weights weights/${basename}.pt --simplify --iou-thres 0.2 --conf-thres 0.01 --img-size 192 192
    python onnx_head_remover.py --type mobilenet_v2_192 --model-name ${basename}_no_opt_192_192.onnx

    conda deactivate
    conda activate onnx2tf
    onnx2tf -i weights/modified_${basename}_no_opt_192_192.onnx -oiqt

    conda deactivate
    conda activate yolov7
    python tflite_runner2.py --weights saved_model/modified_${basename}_no_opt_192_192_integer_quant.tflite \
        --weights2 ../onnx2tf/saved_model/NMS_mobilenet_v2_s192_192_float32.tflite \
        --source ../data/n78_tel8070_application.mp4 --save ${basename}_s192_tel_c025.mp4 --conf 0.25
done