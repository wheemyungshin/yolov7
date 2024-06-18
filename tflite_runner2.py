import numpy as np
import cv2
import os
import tensorflow as tf
import skimage.io


COLORS = [[255,255,0], [255,0,255], [0,255,255]]

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

    predictions = np.squeeze(output)

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
    fd_model = tf.lite.Interpreter("../onnx2tf/saved_model/M_gnet_shufflenet-t_v3_large448_e299_128_256_float32.tflite")
    #nms_part = tf.lite.Interpreter("weights_n78_tflite_nms_sep/nms_float32.tflite")
    fd_model.allocate_tensors()
    #nms_part.allocate_tensors()

    cap = cv2.VideoCapture("../data/gnet_errors1/noperson2_right.mp4")

    vid_name = 'gnet_M_05_tf24.mp4'
    vid_writer = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
    frame_id = 0
    voting_que = [0] * 60
    voting_idx = 0
    while True :
        _, frame = cap.read()

        if frame is not None:
            #frame = frame[::-1, :, :]#.transpose(1, 0, 2)
            frame_vis = frame.copy()
            
            print(fd_model.get_input_details()[0]["shape"])
            #print(nms_part.get_input_details()[0]["shape"])
            crop_img = cv2.resize(frame, (fd_model.get_input_details()[0]["shape"][2], fd_model.get_input_details()[0]["shape"][1]))
            crop_img = (crop_img / 255).astype(np.float32)
            print(np.min(crop_img),np.max(crop_img))
            crop_img = np.expand_dims(crop_img.astype(np.float32), axis=0)
            fd_model.set_tensor(fd_model.get_input_details()[0]['index'], crop_img)
            fd_model.invoke()

            fd_output_0 = fd_model.get_tensor(fd_model.get_output_details()[0]['index'])[0]

            print(fd_output_0)
            print(fd_output_0.shape)

            boxes, scores = nms(fd_output_0, conf_thres=0.5, iou_thres=0.45)
            
            classes = fd_output_1[fd_output_1[:, -1] > opt.conf, 5]
            boxes = fd_output_1[fd_output_1[:, -1] > opt.conf, 1:5]
            scores = fd_output_1[fd_output_1[:, -1] > opt.conf, -1]

            if len(scores) > 0:
                voting_que[voting_idx] = 1#max(scores)
            else:
                voting_que[voting_idx] = 0
            voting_idx+=1
            if voting_idx >= len(voting_que):
                voting_idx = 0
            max_fd = None
            max_size = -1

            print(boxes)
            for idx in range(len(scores)) :
                if scores[idx] > 0.1 :
                    cls_ = int(classes[idx])
                    size = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
                    max_fd = boxes[idx]
                    max_fd[0] = int(max_fd[0] * (480 / fd_model.get_input_details()[0]["shape"][2]))
                    max_fd[2] = int(max_fd[2] * (480 / fd_model.get_input_details()[0]["shape"][2]))
                    max_fd[1] = int(max_fd[1] * (480 / fd_model.get_input_details()[0]["shape"][1]))
                    max_fd[3] = int(max_fd[3] * (480 / fd_model.get_input_details()[0]["shape"][1]))
                    max_fd = max_fd.astype(np.int32)
                    frame_vis = cv2.rectangle(frame_vis, (max_fd[0],max_fd[1]), (max_fd[2],max_fd[3]), COLORS[cls_], 2)            
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

cv2.destroyAllWindows()
print(vid_name)