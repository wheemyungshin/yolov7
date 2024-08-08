import numpy as np
import cv2
import os
import tensorflow as tf
import skimage.io
import argparse

colors = [[20,20,255], [20,255,20], [255,20,20], [255,20,255], [20,255,255], [255,255,20]]

def output_matching(model2_input, fd_outputs):    
    if model2_input["shape"][1] == fd_outputs[0].shape[1]:
        fd_output = fd_outputs[0]
    elif model2_input["shape"][1] == fd_outputs[1].shape[1]:
        fd_output = fd_outputs[1]
    elif model2_input["shape"][1] == fd_outputs[2].shape[1]:
        fd_output = fd_outputs[2]
    else:
        print("wrong input shape: ", model2_input.shape)
    
    return fd_output

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou


def nms(output, conf_thres=0.5, iou_thres=0.45):

    predictions = np.squeeze(output[0])

    # Filter out object confidence scores below threshold

    obj_conf = predictions[:, 4]
    predictions = predictions[obj_conf > conf_thres]
    obj_conf = obj_conf[obj_conf > conf_thres]

    # Multiply class confidence with bounding box confidence
    predictions[:, 5:] *= obj_conf[:, np.newaxis]


    # Get the scores
    scores = np.max(predictions[:, 5:], axis=1)

    # Filter out the objects with a low score
    predictions = predictions[scores > conf_thres]
    scores = scores[scores > conf_thres]

    if len(scores) == 0:
        return [], []

    # Get bounding boxes for each object
    boxes = predictions[:, :4]
    boxes = xywh2xyxy(boxes)

    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    # Select Bbox
    indices = []
    while sorted_indices.size > 0:

        # Pick the last box
        box_id = sorted_indices[0]
        indices.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_thres)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return boxes[indices], scores[indices]

def merge_overlapping_boxes(boxes, scores, overlap_num_thr=5):
    valid_idx = []
    valid_bbox = []
    for b_idxm, box in enumerate(boxes):
        is_invalid = False
        ious = compute_iou(box, boxes)
        for valid_i in valid_idx:
            if ious[valid_i] > 0:
                is_invalid = True
        if not is_invalid:
            overlaps = len(ious[ious > 0])
            print(overlaps)
            if overlaps >= overlap_num_thr:
                valid_idx.append(b_idxm)
                print(boxes[ious > 0])
                merged_bbox = [np.mean(boxes[ious > 0, 0]), np.mean(boxes[ious > 0, 1]), np.mean(boxes[ious > 0, 2]), np.mean(boxes[ious > 0, 3]), np.sum(scores[ious > 0])]
                valid_bbox.append(merged_bbox)
    return np.array(valid_bbox)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../onnx2tf/saved_model/modified_phone_mobilenet_n78_tuning_only3_e089_no_opt_128_128_integer_quant.tflite', help='initial weights path')
    parser.add_argument('--weights2', type=str, default='../onnx2tf/saved_model/NMS_mobilenet_s128_128_float32.tflite', help='initial weights path')
    
    parser.add_argument('--weights3', type=str, default='../onnx2tf/saved_model/modified_phone_mobilenet_n78_tuning_only3_e089_no_opt_128_128_integer_quant.tflite', help='initial weights path')
    parser.add_argument('--weights4', type=str, default='../onnx2tf/saved_model/NMS_mobilenet_s128_128_float32.tflite', help='initial weights path')

    parser.add_argument('--source', type=str, default='../data/n78_tel8070_application.mp4', help='initial weights path')
    parser.add_argument('--save', type=str, default='n78_tuning3_phone_s128_e089_tel_c05.mp4', help='initial weights path')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')

    opt = parser.parse_args()
    
    fd_model = tf.lite.Interpreter(opt.weights)    
    fd_model2= tf.lite.Interpreter(opt.weights2)
    fd_model.allocate_tensors()
    fd_model2.allocate_tensors()

    fd_model3 = tf.lite.Interpreter(opt.weights3)    
    fd_model4 = tf.lite.Interpreter(opt.weights4)
    fd_model3.allocate_tensors()
    fd_model4.allocate_tensors()

    cap = cv2.VideoCapture(opt.source)

    vid_name = opt.save
    vid_writer = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1296, 1080))
    frame_id = 0
    voting_que = [0] * 60
    voting_idx = 0
    while True :
        _, frame = cap.read()

        if frame is not None:
            frame = frame[:, 312:1608]
            #frame = frame[:, :, :]#.transpose(1, 0, 2)
            frame_vis = frame.copy()

            print(fd_model.get_input_details()[0]["shape"])
            crop_img = cv2.resize(frame, (fd_model.get_input_details()[0]["shape"][2], fd_model.get_input_details()[0]["shape"][1]))
            crop_img = (crop_img / 255).astype(np.float32)
            print(np.min(crop_img),np.max(crop_img))
            crop_img = np.expand_dims(crop_img.astype(np.float32), axis=0)

            # vehicle detection
            fd_model.set_tensor(fd_model.get_input_details()[0]['index'], crop_img)
            fd_model.invoke()

            fd_output_0_0_ = fd_model.get_tensor(fd_model.get_output_details()[0]['index'])
            fd_output_0_1_ = fd_model.get_tensor(fd_model.get_output_details()[1]['index'])
            fd_output_0_2_ = fd_model.get_tensor(fd_model.get_output_details()[2]['index'])

            fd_output_0_0 = output_matching(fd_model2.get_input_details()[0], [fd_output_0_0_, fd_output_0_1_, fd_output_0_2_])
            fd_output_0_1 = output_matching(fd_model2.get_input_details()[1], [fd_output_0_0_, fd_output_0_1_, fd_output_0_2_])
            fd_output_0_2 = output_matching(fd_model2.get_input_details()[2], [fd_output_0_0_, fd_output_0_1_, fd_output_0_2_])

            fd_model2.set_tensor(fd_model2.get_input_details()[0]['index'], fd_output_0_0)
            fd_model2.set_tensor(fd_model2.get_input_details()[1]['index'], fd_output_0_1)
            fd_model2.set_tensor(fd_model2.get_input_details()[2]['index'], fd_output_0_2)
            fd_model2.invoke()

            fd_output_1 = fd_model2.get_tensor(fd_model2.get_output_details()[0]['index'])            

            # person detection            
            fd_model3.set_tensor(fd_model3.get_input_details()[0]['index'], crop_img)
            fd_model3.invoke()

            fd_output_3_0_ = fd_model3.get_tensor(fd_model3.get_output_details()[0]['index'])
            fd_output_3_1_ = fd_model3.get_tensor(fd_model3.get_output_details()[1]['index'])
            fd_output_3_2_ = fd_model3.get_tensor(fd_model3.get_output_details()[2]['index'])

            fd_output_3_0 = output_matching(fd_model4.get_input_details()[0], [fd_output_3_0_, fd_output_3_1_, fd_output_3_2_])
            fd_output_3_1 = output_matching(fd_model4.get_input_details()[1], [fd_output_3_0_, fd_output_3_1_, fd_output_3_2_])
            fd_output_3_2 = output_matching(fd_model4.get_input_details()[2], [fd_output_3_0_, fd_output_3_1_, fd_output_3_2_])

            fd_model4.set_tensor(fd_model4.get_input_details()[0]['index'], fd_output_3_0)
            fd_model4.set_tensor(fd_model4.get_input_details()[1]['index'], fd_output_3_1)
            fd_model4.set_tensor(fd_model4.get_input_details()[2]['index'], fd_output_3_2)
            fd_model4.invoke()

            fd_output_3 = fd_model4.get_tensor(fd_model4.get_output_details()[0]['index'])
            
            print(fd_output_1.shape)
                        
            boxes1 = fd_output_1[fd_output_1[:, -1] > opt.conf, 1:5]
            scores1 = fd_output_1[fd_output_1[:, -1] > opt.conf, -1]
            cls_ids1 = fd_output_1[fd_output_1[:, -1] > opt.conf, 5]
            cls_ids1 = cls_ids1+1

            boxes3 = fd_output_3[fd_output_3[:, -1] > opt.conf, 1:5]
            scores3 = fd_output_3[fd_output_3[:, -1] > opt.conf, -1]
            cls_ids3 = fd_output_3[fd_output_3[:, -1] > opt.conf, 5]

            if len(scores1) and len(scores3):
                scores = np.concatenate((scores1, scores3), axis=0)
                boxes = np.concatenate((boxes1, boxes3), axis=0)
                cls_ids = np.concatenate((cls_ids1, cls_ids3), axis=0)
            elif len(scores1):
                scores = scores1
                boxes = boxes1
                cls_ids = cls_ids1
            else:
                scores = scores3
                boxes = boxes3
                cls_ids = cls_ids3

            
            print(scores)
            print(boxes)
            print(cls_ids)
            
            #if len(boxes) > 0:
            #    boxes = merge_overlapping_boxes(boxes, scores, overlap_num_thr=0)
            #
            #if len(boxes) > 0:
            #    scores = boxes[:, -1]
            #    print(scores)
            #    print(boxes)
            #else:
            #    scores = []


            if len(scores) > 0:
                voting_que[voting_idx] = 1#max(scores)
            else:
                voting_que[voting_idx] = 0
            voting_idx+=1
            if voting_idx >= len(voting_que):
                voting_idx = 0
            max_fd = None
            max_size = -1

            for idx in range(len(scores)) :
                if scores[idx] > 0.1 :
                    size = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
                    max_fd = boxes[idx]
                    max_fd[0] = int(max_fd[0] * (1296 / fd_model.get_input_details()[0]["shape"][2]))
                    max_fd[2] = int(max_fd[2] * (1296 / fd_model.get_input_details()[0]["shape"][2]))
                    max_fd[1] = int(max_fd[1] * (1080 / fd_model.get_input_details()[0]["shape"][1]))
                    max_fd[3] = int(max_fd[3] * (1080 / fd_model.get_input_details()[0]["shape"][1]))
                    max_fd = max_fd.astype(np.int32)
                    frame_vis = cv2.rectangle(frame_vis, (max_fd[0],max_fd[1]), (max_fd[2],max_fd[3]), colors[int(cls_ids[idx])], 2)
                    cv2.putText(frame_vis, str(round(scores[idx], 5)), (max_fd[0],max_fd[1] - 2), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            
            voting_score = sum(voting_que) / len(voting_que) 
            if voting_score > 0.4:
                cv2.putText(frame_vis, str(round(voting_score, 5)), (10, 30), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
            else:
                cv2.putText(frame_vis, str(round(voting_score, 5)), (10, 30), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            vid_writer.write(frame_vis)
        else:
            break
        frame_id += 1

    vid_writer.release()

print(vid_name)