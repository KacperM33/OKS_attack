import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import json
import torch
from tabulate import tabulate
import os
import matplotlib.pyplot as plt
import scipy.io
import sys

# random seed
tools_dir = "../oks_tools"
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from utils_seed import set_deterministic_seed

set_deterministic_seed(42)

def OKS_score(ground_keypoints, target_keypoints, sigmas=None, area=None):
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

    if len(ground_keypoints) == 0 or len(target_keypoints) == 0:
        return 0.0

    gt = np.squeeze(ground_keypoints) # Shape (17,2)
    target = np.squeeze(target_keypoints) # Shape (17,2)

    # squared Euclidean distance
    dist_sq = (target[:,0] - gt[:,0]) ** 2 + (target[:,1] - gt[:,1])**2

    # target visibility mask
    vis_mask = np.ones(len(target), dtype=bool)
    
    # area from COCO validation data (if not specified otherwise)
    if area is None:
        area = 30699.56495

    # standard COCO sigmas (if not specified otherwise)
    if sigmas is None:
        sigmas = np.array([.26, .79, .79, .72, .72, 
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


def match_and_score(ground_instances, target_instances, bbox=None):
    """
    Matches persons between predictions and ground truth targets. 
    (Note: Actual matching is currently bypassed; it simply assumes that target[0] corresponds to pred[0]).

    Provides a list of (pred_idx, target_idx) pairs.
    Returns the average score across the detected persons.
    """
    okss = []
    losses = 0
    for idx in range(0, (len(ground_instances))):
        ground_frame = ground_instances[idx]
        target_frame = target_instances[idx]

        is_gt_empty = np.isnan(ground_frame).any()
        is_target_empty = np.isnan(target_frame).any()

        # area calculated based on bounding boxes
        if bbox is not None:
            width = bbox[idx][2] - bbox[idx][0]
            height = bbox[idx][3] - bbox[idx][1]
            area = float(width * height)
        else:
            area = None 

        if is_gt_empty or is_target_empty:
            if is_target_empty and not is_gt_empty:
                losses += 1
            okss.append(0.0)
        else:
            final_oks = OKS_score(ground_frame, target_frame, area=area)

            okss.append(final_oks)
    
    return np.mean(okss), losses


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

    frames_keypoints_fixed = []
    idx_to_del = [1, 2, 3, 4] # left/right eye, left/right ear

    for persons in frames_keypoints:
        if len(persons) == 0:
            dummy_person = np.full((13, 2), np.nan)
            frames_keypoints_fixed.append(dummy_person)
        else:
            person_arr = np.array(persons[0])
            person_arr = np.delete(person_arr, idx_to_del, axis=0)
            frames_keypoints_fixed.append(person_arr)

    frames_keypoints_array = np.array(frames_keypoints_fixed)

    return frames_keypoints_array


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

        if len(persons) == 0:
            persons.append([np.nan, np.nan, np.nan, np.nan])
            
        frames_bboxes.append(persons)

    return np.squeeze(frames_bboxes)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--ground',
        type=str,
        default='',
        help='ground truth keypoints')
    parser.add_argument(
        '--test_org',
        type=str,
        default='',
        help='original video test keypoints')
    parser.add_argument(
        '--test_adv',
        type=str,
        default='',
        help='adversarial video test keypoints')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='output path')
    parser.add_argument(
        '--model',
        type=str,
        default='???',
        help='model name')
    parser.add_argument(
        '--test',
        type=str,
        default='???',
        help='test name')

    args = parser.parse_args()

    if args.ground == '':
        raise ValueError('Ground truth must not be empty, use --ground path_to_ground_truth_keypoints.mat')
    else:
        ground_path = Path(args.ground)

    model_name = args.model
    test_name = args.test

    if args.test_adv == '' and args.test_org == '':
        raise ValueError('Atleast one orginal or adversarial video must not be empty, use --test_org or --test_adv path_to_video_keypoints.json')
    else:
        if args.test_adv != '':
            adv_path = Path(args.test_adv)
        else:
            adv_path = None
        if args.test_org != '':
            org_path = Path(args.test_org)
        else:
            org_path = None
    if args.output == '':
        raise ValueError('Output path must not be empty, use --output path_to_output_folder')
    else:
        output_path = Path(args.output)

    ground_truth_mat = scipy.io.loadmat(ground_path)

    ground_truth_raw = np.stack((ground_truth_mat['x'], ground_truth_mat['y']), axis=-1)
    
    # MMPose keypoints are ordered left/right/left/right
    # Penn Action ground truth keypoints are ordered right/left/right/left
    # therefore, their order needs to be swapped
    correct_order = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]

    ground_truth = ground_truth_raw[:, correct_order, :]
    
    if adv_path is None:
        print(f"adversarial video keypoints empty")
    else:
        advers_kp = load_keypoints_from_json(adv_path)
        advers_bbox = load_bbox_from_json(adv_path)
    if org_path is None:
        print(f"original video keypoints empty")
    else:
        origin_kp = load_keypoints_from_json(org_path)
        origin_bbox = load_bbox_from_json(org_path)
    

    if 'origin_kp' in locals():
        print("\nCalculating OKS_score for the original video...")
        score_org, loss_org = match_and_score(ground_truth, origin_kp, origin_bbox)
        print(f"RESULT: {score_org:.2f}")

    if 'advers_kp' in locals():
        print("\nCalculating OKS_score for the adversarial video...")
        score_adv, loss_adv = match_and_score(ground_truth, advers_kp, advers_bbox)
        print(f"RESULT: {score_adv:.2f}")

    data = [
        ["", "Mean OKS score", f'{score_org:.2f}', f'{score_adv:.2f}'],
        [f"{model_name}", "Losses", loss_org, loss_adv],
        [f"{test_name}", "", "", ""],
        ["", "Difference", "", f'{round(score_org - score_adv, 2):.2f}'],
    ]

    headers = ["Test", "Measure", "Original", "Adversarial"]

    print()
    print(tabulate(data, headers=headers, tablefmt="grid"))

    # saving as an image
    fig, ax = plt.subplots(figsize=(5, 1.5)) 
    ax.axis('tight')
    ax.axis('off')

    tab = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.auto_set_column_width(col=list(range(len(headers))))
    tab.scale(1, 1.5)

    # Top cell in the column - Left (L), Right (R), and Top (T) edges
    tab[1, 0].visible_edges = 'LRT'
    # Middle cells - Left (L) and Right (R) edges
    tab[2, 0].visible_edges = 'LR'
    tab[3, 0].visible_edges = 'LR'
    # Bottom cell - Left (L), Right (R), and Bottom (B) edges
    tab[4, 0].visible_edges = 'LRB'


    # Row 3 cell formatting - setting specific borders for columns 1-3
    tab[3, 1].visible_edges = 'LTB'
    tab[3, 2].visible_edges = ''
    tab[3, 3].visible_edges = 'RTB'

    # Row 4 cell formatting - setting specific borders for columns 2-3
    tab[4, 2].visible_edges = 'LTB'
    tab[4, 3].visible_edges = 'RTB'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(f"{output_path}/results.png", bbox_inches='tight', dpi=300)

    print("Table saved as results.png!")

if __name__ == '__main__':
    main()