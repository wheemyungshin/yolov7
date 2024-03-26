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


if __name__ == '__main__':

    fd_model = tf.lite.Interpreter("/data/gnet_tflite_test_models/A_adam_try_crowd_yolov7-shufflenet-t_s448_e060_128_224_float32.tflite")
    fd_model.allocate_tensors()

    for img_name_id in range(426):
        frame = np.load(os.path.join('/data/gnet_live_samples', str(img_name_id)+'.npy'))
        frame = np.reshape(frame, (464 , 800 , 3))
        frame = frame[:, :, [2,1,0]]

        frame_vis = frame.copy()

        input_h = fd_model.get_input_details()[0]["shape"][1]
        input_w = fd_model.get_input_details()[0]["shape"][2]

        '''
        resize_ratio = (input_w / frame.shape[1])
        crop_img = cv2.resize(frame, (int(frame.shape[1] * resize_ratio), int(frame.shape[0] * resize_ratio)))

        img_board = (np.ones((input_h, input_w, 3)) * (114, 114, 114)).astype(np.uint8)
        input_resize_gap = int((input_h - crop_img.shape[0])/2)
        img_board[input_resize_gap:-input_resize_gap, :, :] = crop_img
        '''
        img_board = cv2.resize(frame, (224, 128))

        img_board = (img_board / 255).astype(np.float32)
        img_board = np.expand_dims(img_board.astype(np.float32), axis=0)
        fd_model.set_tensor(fd_model.get_input_details()[0]['index'], img_board)
        fd_model.invoke()

        fd_output_0 = np.load(os.path.join('/data/gnet_live_samples_npy_inout', str(img_name_id)+'_000000_clean.jpgoutput.npy'))
        fd_output_0 = np.expand_dims(fd_output_0, 0)
        print(fd_output_0.shape)
        #fd_output_0 = fd_model.get_tensor(fd_model.get_output_details()[0]['index'])
        boxes, scores = nms(fd_output_0)

        max_fd = None
        max_size = -1

        for idx in range(len(scores)) :
            if scores[idx] > 0.3 :
                size = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])
                max_fd = boxes[idx]
                max_fd[0] = int(max_fd[0] * ( frame.shape[1] / 192))
                max_fd[2] = int(max_fd[2] * ( frame.shape[1] / 96))
                max_fd[1] = int(max_fd[1] * ( frame.shape[0] / 192))
                max_fd[3] = int(max_fd[3] * ( frame.shape[0] / 96))
                max_fd = max_fd.astype(np.int32)
                frame_vis = cv2.rectangle(frame_vis, (max_fd[0],max_fd[1]), (max_fd[2],max_fd[3]), (255,255,0), 2)            
                cv2.putText(frame_vis, str(round(scores[idx], 5)), (max_fd[0],max_fd[1] - 2), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                print(str(img_name_id)+'.jpg', " : ", scores[idx])

        skimage.io.imsave("demo_"+str(img_name_id)+'.jpg', frame_vis)


