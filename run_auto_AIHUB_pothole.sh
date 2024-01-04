python3 detect.py --weights weights/cityscapes_A2D2_waymo_BDD100K_yolov7x_s576_960_smallest_nomosaic_rot_and_more_lrtune.pt --conf 0.55 --iou-thres 0.65 --source /data/AIHUB_bicycle_obstacle_potholes/images/val --name auto_AIHUB_pothole_val --img-size 960 576 --device 4 --agnostic-nms --save-txt --save-frame --classes 2 4 6 7 8 10 11
python3 detect.py --weights weights/cityscapes_A2D2_waymo_BDD100K_yolov7x_s576_960_smallest_nomosaic_rot_and_more_lrtune.pt --conf 0.55 --iou-thres 0.65 --source /data/AIHUB_bicycle_obstacle_potholes/images/train --name auto_AIHUB_pothole_train --img-size 960 576 --device 4 --agnostic-nms --save-txt --save-frame --classes 2 4 6 7 8 10 11