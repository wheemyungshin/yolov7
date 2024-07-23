import numpy as np
import cv2
import os
import tensorflow as tf


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


if __name__ == '__main__':

    fd_model = tf.lite.Interpreter("weights_n78_tflite/BC_phone_mobilenet_manual_resize_range16_64_lessrot_128_128_integer_quant.tflite")
    fd_model.allocate_tensors()


    cap = cv2.VideoCapture("right_01.mp4")

    while True :
        _, frame = cap.read()

        frame_vis = frame.copy()

        crop_img = cv2.resize(frame, (fd_model.get_input_details()[0]["shape"][2], fd_model.get_input_details()[0]["shape"][1]))
        crop_img = (cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB) / 255).astype(np.float32)
        crop_img = np.expand_dims(crop_img.astype(np.float32), axis=0)
        fd_model.set_tensor(fd_model.get_input_details()[0]['index'], crop_img)
        fd_model.invoke()

        fd_output_0 = fd_model.get_tensor(fd_model.get_output_details()[0]['index'])
        boxes, scores = nms(fd_output_0)

        max_fd = None
        max_size = -1

        for idx in range(len(scores)) :
            if scores[idx] > 0.3 :
                size = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
                max_fd = boxes[idx]
                max_fd[0] = int(max_fd[0] * ( frame.shape[1] / fd_model.get_input_details()[0]["shape"][2]))
                max_fd[2] = int(max_fd[2] * ( frame.shape[1] / fd_model.get_input_details()[0]["shape"][2]))
                max_fd[1] = int(max_fd[1] * ( frame.shape[0] / fd_model.get_input_details()[0]["shape"][1]))
                max_fd[3] = int(max_fd[3] * ( frame.shape[0] / fd_model.get_input_details()[0]["shape"][1]))
                max_fd = max_fd.astype(np.int32)
                frame_vis = cv2.rectangle(frame_vis, (max_fd[0],max_fd[1]), (max_fd[2],max_fd[3]), (255,255,0), 2)            

        cv2.imshow("demo", frame_vis)
        key = cv2.waitKey(1)


cv2.destroyAllWindows()

