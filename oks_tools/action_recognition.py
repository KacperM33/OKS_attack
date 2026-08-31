import numpy as np
import os
import json
import torch
import glob
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from tabulate import tabulate
import scipy.io
import sys

from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.dataset import Compose, pseudo_collate
from mmengine.runner import load_checkpoint

from mmaction.utils import register_all_modules
register_all_modules()

# random seed
tools_dir = "../oks_tools"
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from utils_seed import set_deterministic_seed

set_deterministic_seed(42)

def kps_action_recognition(input_path):
    """
    Function to run action recognition
    """
    # model config
    config_file = '../mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb32-u48-240e_k400-keypoint.py'

    # model weights
    checkpoint_file = '../mmaction2/checkpoints/slowonly_r50_k400-keypoint.pth'

    cfg = Config.fromfile(config_file)

    # building a clean, raw PyTorch model
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.eval()

    # extracting the data pipeline to transform the dictionary into tensors
    pipeline = Compose(cfg.test_dataloader.dataset.pipeline)

    json_files = glob.glob(os.path.join(input_path, "*cleaned.json"))
    if not json_files:
        json_files = glob.glob(os.path.join(input_path, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in directory {input_path}")
    
    pose_data_path = json_files[0]
    pose_data_name = os.path.splitext(os.path.basename(pose_data_path))[0]

    print(f"\nFound JSON file: {pose_data_name}")

    with open(pose_data_path, 'r') as f:
        data = json.load(f)

    frames = data['instance_info']
    T = len(frames)

    keypoint_list = []
    frame_inds_list = []

    for i, frame in enumerate(frames):
        instances = frame['instances']
        if not instances:
            continue

        # idx 0 - one person on video
        target_instance = instances[0]
        kps = np.array(target_instance['keypoints'], dtype=np.float32)
        scores = np.array(target_instance['keypoint_scores'], dtype=np.float32)[:, None]

        kp_3d = np.concatenate([kps, scores], axis=1)

        keypoint_list.append(kp_3d)
        frame_inds_list.append(i) # frame idx

    # constructing the input data dictionary in the format expected by the OpenMMLab pipeline
    data_dict = dict(
        frame_dir=pose_data_name,
        label=-1,                               # -1 indicates dummy label for inference
        img_shape=(480, 640),
        original_shape=(480, 640),
        start_index=0,
        modality='Pose',
        total_frames=T,
        frame_inds=np.array(frame_inds_list),   # modality set for skeleton-based action recognition
        keypoint=np.array(keypoint_list),
    )

    data_transformed = pipeline(data_dict)
    
    # collating the single transformed sample into a batch for model inference
    data_batch = pseudo_collate([data_transformed])

    # inference
    print(f"\nPerforming action recognition for {pose_data_name}...")
    with torch.no_grad():
        outputs = model.test_step(data_batch)

    print("\n--- SUCCESS ---")
    
    return outputs


def top3_classes_kps(outputs):
    """
    Function to save the top 3 recognized classes (based on the Kinetics-400 / k400 dataset).
    """
    # path to the labels file
    label_file = '../mmaction2/tools/data/kinetics/label_map_k400.txt'

    classes = []
    if os.path.exists(label_file):
        with open(label_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines()]
    else:
        print(f"WARNING: File {label_file} not found")

    pred_scores = outputs[0].pred_score.cpu().numpy()

    # extracting the top 3 predictions
    top3_indices = np.argsort(pred_scores)[::-1][:3]
    top3_dict = {}

    for i, idx in enumerate(top3_indices):
        # extracting the class name if the file was loaded and the index exists
        if classes and idx < len(classes):
            action_name = classes[idx]
        else:
            action_name = "Brak nazwy w pliku"

        top3_dict[i] =  {
            'idx': f'{idx}',
            'action_name': f'{action_name}',
            'pred_score': f'{pred_scores[idx]*100:.2f}'
        }

    return top3_dict


def print_res(top3_class):
    print()
    for i in top3_class:
        print(f"Top {i+1}: Class {top3_class[i]['idx']} - {top3_class[i]['action_name']} | {top3_class[i]['pred_score']}%")
    print("="*50)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--exp',
        type=str,
        default='',
        help='experiment name of folder with exp')
    parser.add_argument(
        '--og-kpts',
        type=str,
        default='',
        help='folder with json with original keypoints')
    parser.add_argument(
        '--adv-kpts',
        type=str,
        default='',
        help='folder with json with adversarial keypoints')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='output path folder')
    parser.add_argument(
        '--gt',
        type=str,
        default='',
        help='groundtruth file path')
    
    
    args = parser.parse_args()

    if args.exp == '':
        raise ValueError('Name of experiment (folder with) is required, use --exp experiment_name')
    else:
        experiment_name = f'{args.exp}'

    if args.og_kpts == '':
        raise ValueError('Folder with original keypoints json is required, use --og-kpts path_to_folder_with_json')
    else:
        og_keypoints_path = f'{args.og_kpts}'

    if args.adv_kpts == '':
        raise ValueError('Folder with adversarial keypoints json is required, use --adv-kpts path_to_folder_with_json')
    else:
        adv_keypoints_path = f'{args.adv_kpts}'

    if args.output == '':
        raise ValueError('Output folder is required, use --output path_to_output_folder')
    else:
        output_path = f'{args.output}'

    if args.gt == '':
        groundtruth = ''
    else:
        groundtruth = f'{args.gt}'
        groundtruth_mat = scipy.io.loadmat(groundtruth)
    
    print(f"Running Action Recognition for {experiment_name}")

    # original
    kps_outputs_og = kps_action_recognition(og_keypoints_path)
    top3_class_og = top3_classes_kps(kps_outputs_og)
    print_res(top3_class_og)

    # adversarial
    kps_outputs_adv1 = kps_action_recognition(adv_keypoints_path)
    top3_class_adv1 = top3_classes_kps(kps_outputs_adv1)
    print_res(top3_class_adv1)

    diff = []

    if str(top3_class_og[0]['action_name']) == str(top3_class_adv1[0]['action_name']):
        val = round((float(top3_class_og[0]['pred_score']) - float(top3_class_adv1[0]['pred_score'])), 2)
        diff.append(val)
    else:
        diff.append('Different')

    data = [
        *([["Groundtruth", '-', groundtruth_mat['action'][0], '-', '-']] if groundtruth != '' else []),
        ["Original", top3_class_og[0]['idx'], top3_class_og[0]['action_name'], f"{top3_class_og[0]['pred_score']}%", '-']
    ]

    
    data.append([f"Adversarial", top3_class_adv1[0]['idx'], top3_class_adv1[0]['action_name'], f"{top3_class_adv1[0]['pred_score']}%", diff[0]])

    headers = ["Video", "Class ID", "Class", "Confidence", "Difference"]

    print()
    print(tabulate(data, headers=headers, tablefmt="grid"))

    # saving as an image
    fig, ax = plt.subplots(figsize=(1, 1)) 
    ax.axis('tight')
    ax.axis('off')

    tab = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.auto_set_column_width(col=list(range(len(headers))))

    tab.scale(1, 3)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(f"{output_path}/action_recognition.png", bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()