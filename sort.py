from __future__ import print_function
import numpy as np
from numpy import zeros, eye
import cv2

np.random.seed(0)

"""From SORT: Computes IOU between two boxes in the form [x1,y1,x2,y2]"""
def iou_batch(bb_test, bb_gt):    
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[...,0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


"""Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form [x,y,s,r] where x,y is the center of the box and s is the scale/area and r is the aspect ratio"""
def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = float(int(w)) * float(int(h))
    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


"""Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right"""
def convert_x_to_bbox(x, score=None): 
    w = np.sqrt(abs(x[2] * x[3]))        
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

"""This class represents the internal state of individual tracked objects observed as bbox."""
class KalmanBoxTracker(object):
    
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box
        
        Parameter 'bbox' must have 'detected class' int number at the -1 position.
        """
        self.kf = cv2.KalmanFilter(7, 4, 0)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], dtype=np.float64)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], dtype=np.float64)
        self.kf.processNoiseCov = np.eye(7)
        self.kf.measurementNoiseCov = eye(4)
        self.kf.errorCovPost = np.eye((7))
        
        self.state = zeros((7, 1))
        self.kf.statePost = self.state.copy()

        self.kf.measurementNoiseCov[2:,2:] *= 10.
        self.kf.errorCovPost[4:,4:] *= 1000.
        self.kf.errorCovPost *= 10.
        self.kf.processNoiseCov[-1,-1] *= 0.5
        self.kf.processNoiseCov[4:,4:] *= 0.5
        
        self.kf.statePost[:4] = convert_bbox_to_z(bbox) # STATE VECTOR 
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.centroidarr = []
        CX = (bbox[0]+bbox[2])//2
        CY = (bbox[1]+bbox[3])//2
        self.centroidarr.append((CX,CY))
        
        #keep yolov5 detected class information
        self.detclass = bbox[5]
        
    def update(self, bbox):
        """
        Updates the state vector with observed bbox
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.correct(convert_bbox_to_z(bbox))

        self.detclass = bbox[5]
        CX = (bbox[0]+bbox[2])//2
        CY = (bbox[1]+bbox[3])//2
        self.centroidarr.append((CX,CY))
    
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate
        """
        if((self.state[6]+self.state[2])<=0):
            self.state[6] *= 0.0
        self.state = self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        
        return convert_x_to_bbox(self.state)
    
    
    def get_state(self):
        """
        Returns the current bounding box estimate
        # test
        arr1 = np.array([[1,2,3,4]])
        arr2 = np.array([0])
        arr3 = np.expand_dims(arr2, 0)
        np.concatenate((arr1,arr3), axis=1)
        """
        arr_detclass = np.expand_dims(np.array([self.detclass]), 0)
        
        return np.concatenate((convert_x_to_bbox(self.kf.statePost), arr_detclass), axis=1)
    
def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of 
    1. matches,
    2. unmatched_detections
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections))
    
    iou_matrix = iou_batch(detections, trackers)
    
    unmatched_detections = []
    matches = []
    if min(iou_matrix.shape) > 0:        
        for i, j in enumerate(np.argmax(iou_matrix, axis=1)):
            if iou_matrix[i, j] > iou_threshold:
                matches.append(np.array([i,j]).reshape(1,2))
            else:
                unmatched_detections.append(i)
    else:
        matches = np.empty(shape=(0,2))

    if(len(matches)==0):
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections)
    

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    def getTrackers(self,):
        return self.trackers
        
    def update(self, dets= np.empty((0,6))):
        """
        Parameters:
        'dets' - a numpy array of detection in the format [[x1, y1, x2, y2, score], [x1,y1,x2,y2,score],...]
        
        Ensure to call this method even frame has no detections. (pass np.empty((0,5)))
        
        Returns a similar array, where the last column is object ID (replacing confidence score)
        
        NOTE: The number of objects returned may differ from the number of objects provided.
        """
        self.frame_count += 1
        dead_trackers = []
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(np.hstack((dets[i,:], np.array([0]))))
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) #+1'd because MOT benchmark requires positive value
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age) and self.max_age >= 0:
                dead_trackers.append(self.trackers.pop(i))
        if(len(ret) > 0):
            return np.concatenate(ret), dead_trackers
        return np.empty((0,6)), dead_trackers