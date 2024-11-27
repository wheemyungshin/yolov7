source ~/.bashrc
conda activate yolov7

target="seatbelt1008_v65s"
for w in $target/*; do
        if [[ "$w" == *epoch*.pt ]]; then
            weight=${w##*/}
            echo "${weight%.*}"

            basename="${weight%.*}"

            python export.py --weights ${target}/${basename}.pt --simplify --iou-thres 0.7 --conf-thres 0.01 --img-size 96 96
            python onnx_head_remover.py --type mobilenet_t_96_96_test --model-name ${basename}_no_opt_96_96.onnx --dir ${target}

            conda deactivate
            conda activate onnx2tf
            onnx2tf -i ${target}/modified_${basename}_no_opt_96_96.onnx -oiqt

            conda deactivate
            conda activate yolov7

            python tflite_runner_for_pittaseatbeltauto.py \
                --weights saved_model/modified_${basename}_no_opt_96_96_float16.tflite \
                --weights2 saved_model/NMS_mobilenet_t_96_96_float32.tflite \
                --source ../data/20241021_132427_YO.mp4 --save forpittatest.mp4 --conf 0.5
        fi
done

