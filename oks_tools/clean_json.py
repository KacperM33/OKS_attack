import json
import numpy as np
import scipy.io
import sys
from argparse import ArgumentParser
from pathlib import Path

# random seed
tools_dir = "../oks_tools"
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from utils_seed import set_deterministic_seed

set_deterministic_seed(42)


def IoUScore(bbox1, bbox2):
    """
    IoU Score - Intersection over Union

    Args:
        bbox1: Bounding boxes (provided in JSON format).
        bbox2: Bounding boxes (provided in JSON format).

    Returns:
        Values in the range [0, 1].
        PERFECT MATCH = 1
        NO MATCH = 0
    """
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bot = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bot < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bot - y_top)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area
    

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--mat',
        type=str,
        default='',
        help='keypoints in .mat file type (ground truth)')
    parser.add_argument(
        '--json',
        type=str,
        default='',
        help='keypoints in .json file type (detector)')

    args = parser.parse_args()

    if args.mat == '':
        raise ValueError('Mat file must not be empty, use --mat path_to_mat_keypoints.mat')
    else:
        path_mat = Path(args.mat)

    if args.json == '':
        raise ValueError('Json file must not be empty, use --json path_to_json_keypoints.json')
    else:
        path_json = Path(args.json)
        
    
    # load .mat file
    data_mat = scipy.io.loadmat(path_mat)

    # load .json file
    with open(path_json, 'r') as f:
        data_json = json.load(f)

    frames = data_json['instance_info']
    iou_threshold = 0.5 
    lost_frames = 0

    for i in range(0, len(frames)):
        inst = frames[i]['instances']

        best_iou = -1.0
        best_inst_idx = -1

        for j in range(0, len(inst)):
            bbox_json = np.array(inst[j]['bbox'][0])
            bbox_mat = np.array(data_mat['bbox'][i])

            iou_score = IoUScore(bbox_json, bbox_mat)

            if iou_score > best_iou:
                best_iou = iou_score
                best_inst_idx = j

        if best_iou >= iou_threshold and best_inst_idx != -1:
            frames[i]['instances'] = [inst[best_inst_idx]]
        else:
            frames[i]['instances'] = [np.nan, np.nan, np.nan, np.nan]
            lost_frames += 1

    output_path = path_json.parent / f"{path_json.stem}_cleaned{path_json.suffix}"
    with open(output_path, 'w') as f:
        json.dump(data_json, f, indent=4)

    print(f'Filtered file saved as {output_path.name}. Number of lost frames: {lost_frames}')

if __name__ == '__main__':
    main()