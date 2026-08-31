import json
import numpy as np
import subprocess
import os

# ============ load json files ============

def load_keypoints_from_json(path):
    """
    Loads a JSON file saved by MMPose and returns a numpy array in the format:
    all_keypoints[frame][person][keypoint] = [x, y]
    """
    with open(path, 'r') as f:
        data = json.load(f)

    frames_keypoints = []
    for frame in data["instance_info"]:
        persons = []
        for inst in frame["instances"]:
            if "keypoints" in inst:
                keypoints = inst["keypoints"]
                if isinstance(keypoints[0], list):
                    coords = [[kp[0], kp[1]] for kp in keypoints]
                else:
                    coords = [keypoints[i:i+2] for i in range(0, len(keypoints), 3)]
                persons.append(coords)
        frames_keypoints.append(persons)

    return frames_keypoints


def load_bbox_from_json(path):
    """
    Loads a JSON file saved by MMPose and returns a numpy array in the format:
    all_bboxes[frame][person][bbox] = [x1, y1, x2, y2]
    """
    with open(path, 'r') as f:
        data = json.load(f)

    frames_bboxes = []
    for frame in data["instance_info"]:
        persons = []
        for inst in frame["instances"]:
            if "bbox" in inst:
                bboxes = inst["bbox"]
                if isinstance(bboxes[0], list):
                    coords = bboxes[0][:4]
                else:
                    coords = bboxes[:4]
                persons.append(coords)
                persons = np.squeeze(persons)

        if len(persons) == 0:
            persons.append(np.empty((0, 4), dtype=np.float32))
            
        frames_bboxes.append(persons)

    return frames_bboxes


# ================ scores ================

def OKS_score(pred_keypoints_json, target_keypoints, area=None, sigmas=None):
    """
    OKS Score - Object Keypoint Similarity.

    Args:
        pred_keypoints_json: Predicted keypoints in JSON format.
        target_keypoints: Ground truth keypoints.
        areas: Calculated bounding box area.

    Returns:
        Values in the range [0, 1].
        PERFECT MATCH = 1
        NO MATCH = 0
    """
    epsilon = np.finfo(np.float32).eps

    pred_keypoints = pred_keypoints_json['keypoints']

    if len(pred_keypoints) == 0 or len(target_keypoints) == 0:
        return 0.0

    pred = np.squeeze(pred_keypoints) # Shape (17,2)
    target = np.squeeze(target_keypoints) # Shape (17,2)

    # squared Euclidean distance
    dist_sq = (target[:,0] - pred[:,0]) ** 2 + (target[:,1] - pred[:,1])**2

    # target visibility mask
    vis_mask = np.ones(len(target), dtype=bool)
    
    # area from COCO validation data (if not specified otherwise)
    if area is None:
        area = 30699.56495

    # standard COCO sigmas (if not specified otherwise)
    if sigmas is None:
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, 
                               .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    # COCO assigns k = 2 * sigma
    k = 2*sigmas

    # denominator
    denom = 2 * (k**2) * (area + epsilon)

    # exponent
    exp_term = dist_sq / denom

    # Object Keypoint Similarity
    oks = (np.exp(-exp_term) * vis_mask).sum(-1) / (vis_mask.sum(-1) + epsilon)

    return oks


def IoU_score(pred_bbox, target_bbox, **kwargs):
    """
    IoU Score - Intersection over Union

    Args:
        pred_bbox: Predicted bounding boxes (provided in JSON format).
        target_bbox: Ground truth bounding boxes.

    Returns:
        Values in the range [0, 1].
        PERFECT MATCH = 1
        NO MATCH = 0
    """
    if hasattr(pred_bbox, 'bboxes'):
        pred_bbox = np.squeeze(pred_bbox['bboxes'])

    if (pred_bbox is not None and len(pred_bbox) == 4) and (target_bbox is not None and len(target_bbox) == 4):
        x_left = np.maximum(target_bbox[0], pred_bbox[0])
        y_top = np.maximum(target_bbox[1], pred_bbox[1])
        x_right = np.minimum(target_bbox[2], pred_bbox[2])
        y_bot = np.minimum(target_bbox[3], pred_bbox[3])
        
        if x_right < x_left or y_bot < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bot - y_top)

        bbox_det = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
        bbox_gt = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])

        union_area = bbox_det + bbox_gt - intersection_area

        if union_area == 0:
            return 0.0

        iou_score = intersection_area / union_area

        return iou_score
    else:
        return 0.0
    