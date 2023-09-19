target="/data/small_ticekt_drivers"

for dir in $target/*; do
        if [ -d "$dir" ]; then
                echo $dir
                echo ${dir##*/}
                python3 detect.py --weights runs/train/cityscapes_AIHUB_ADAS_waymo_kitti_A2D2_BDD100k_yolov7x_s384_6402/weights/epoch_086.pt --conf 0.4 --source $dir --name test_small_ticekt_drivers/${dir##*/} --img-size 384 640 --device 0 --agnostic-nms --save-frame --save-txt
        fi
done