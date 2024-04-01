import numpy as np
import cv2
import os
import tensorflow as tf
import skimage.io


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

    fd_model = tf.lite.Interpreter("top5_ircalibration/CA_phone_mobilenet_manual_resize_range24_96_small96from_BDagain_qat_128_128_00_integer_quant.tflite")
    fd_model2= tf.lite.Interpreter("top5_ircalibration/CA_phone_mobilenet_manual_resize_range24_96_small96from_BDagain_qat_128_128_01_float32.tflite")
    nms_part = tf.lite.Interpreter("weights_n78_tflite_nms_sep/nms_float32.tflite")
    fd_model.allocate_tensors()
    fd_model2.allocate_tensors()
    nms_part.allocate_tensors()

    cap = cv2.VideoCapture("/data/n78_testvid.mp4")

    vid_writer = cv2.VideoWriter('n78_CA_crop_ir_01.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (480, 480))
    frame_id = 0
    unique_confidences = []
    while True :
        _, frame = cap.read()

        if frame is not None:
            #frame = frame[::-1, :, :]#.transpose(1, 0, 2)
            frame_vis = frame.copy()

            print(fd_model.get_input_details()[0]["shape"])
            crop_img = cv2.resize(frame, (fd_model.get_input_details()[0]["shape"][2], fd_model.get_input_details()[0]["shape"][1]))
            crop_img = (crop_img / 255).astype(np.float32)
            print(np.min(crop_img),np.max(crop_img))
            crop_img = np.expand_dims(crop_img.astype(np.float32), axis=0)
            fd_model.set_tensor(fd_model.get_input_details()[0]['index'], crop_img)
            fd_model.invoke()

            fd_output_0_0 = fd_model.get_tensor(fd_model.get_output_details()[0]['index'])
            fd_output_0_1 = fd_model.get_tensor(fd_model.get_output_details()[1]['index'])
            fd_output_0_2= fd_model.get_tensor(fd_model.get_output_details()[2]['index'])

            
            print(fd_output_0_0.shape, fd_model2.get_input_details()[0]["shape"])
            print(fd_output_0_1.shape, fd_model2.get_input_details()[1]["shape"])
            print(fd_output_0_2.shape, fd_model2.get_input_details()[2]["shape"])
            fd_model2.set_tensor(fd_model2.get_input_details()[0]['index'], fd_output_0_0)
            fd_model2.set_tensor(fd_model2.get_input_details()[1]['index'], fd_output_0_1)
            fd_model2.set_tensor(fd_model2.get_input_details()[2]['index'], fd_output_0_2)
            fd_model2.invoke()

            fd_output_1 = fd_model2.get_tensor(fd_model2.get_output_details()[0]['index'])
            
            for unique_c in np.unique(fd_output_1[:,-1]):
                if unique_c not in unique_confidences:
                    unique_confidences.append(unique_c)
            #fd_output_0 = np.expand_dims(fd_output_0, 0)
            #boxes, scores = nms(fd_output_0)

            #boxes = fd_output_0[fd_output_0[:, -1] > 0]
            #scores = fd_output_0[fd_output_0[:, -1] > 0, -1]
            #fd_output_1 = np.expand_dims(fd_output_1[:, 1:], 0)
            print(fd_output_1)
            print(fd_output_1.shape)
            #print(nms_part.get_input_details()[0]['shape'])
            
            #nms_part.set_tensor(nms_part.get_input_details()[0]['index'], np.transpose(fd_output_0, (0,2,1)))
            #nms_part.invoke()
            #nms_output = nms_part.get_tensor(nms_part.get_output_details()[0]['index'])
            
            boxes = fd_output_1[fd_output_1[:, -1] > 0.1, 1:5]
            scores = fd_output_1[fd_output_1[:, -1] > 0.1, -1]
            
            '''
            print(scores)
            print(boxes)
            if len(boxes) > 0:
                boxes = merge_overlapping_boxes(boxes, scores, overlap_num_thr=0)
            
            if len(boxes) > 0:
                scores = boxes[:, -1]
                print(scores)
                print(boxes)
            else:
                scores = []
            '''

            max_fd = None
            max_size = -1

            for idx in range(len(scores)) :
                if scores[idx] > 0.1 :
                    size = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
                    max_fd = boxes[idx]
                    max_fd[0] = int(max_fd[0] * (480 / 128))
                    max_fd[2] = int(max_fd[2] * (480 / 128))
                    max_fd[1] = int(max_fd[1] * (480 / 128))
                    max_fd[3] = int(max_fd[3] * (480 / 128))
                    max_fd = max_fd.astype(np.int32)
                    frame_vis = cv2.rectangle(frame_vis, (max_fd[0],max_fd[1]), (max_fd[2],max_fd[3]), (255,255,0), 2)            
                    cv2.putText(frame_vis, str(round(scores[idx], 5)), (max_fd[0],max_fd[1] - 2), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            vid_writer.write(frame_vis)
        else:
            break
        frame_id += 1

    vid_writer.release()

print("unique_confidences : ", unique_confidences)

cv2.destroyAllWindows()

