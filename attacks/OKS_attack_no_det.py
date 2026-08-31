# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import subprocess
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
import torch

from mmengine.logging import print_log
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

from tools import load_keypoints_from_json, load_bbox_from_json, OKS_score, IoU_score
from attack_config import ATTACK_PARAMS
from attack_core import process_one_image, apply_attack

import scipy.io
import sys

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# random seed
tools_dir = "../oks_tools"
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from utils_seed import set_deterministic_seed

set_deterministic_seed(42)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument(
        '--target-keypoints',
        type=str,
        default='',
        help='root of the target keypoints JSON file. ')
    parser.add_argument(
        '--trial',
        type=str,
        default='False',
        help='Start attack full or trial')
    parser.add_argument(
        '--gt',
        type=str,
        default='',
        help='Mat file with groundtruth of video')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    if args.gt == '':
        print("BBOX based on target .json\n")
        # load target bbox
        if args.target_keypoints == '':
            raise ValueError('target_keypoints must not be empty, use --target-keypoints path_to_target_keypoints.json')
        else:
            target_bboxes = load_bbox_from_json(args.target_keypoints)
    else:
        print("BBOX based on ground truth .mat\n")
        # load .mat file with ground truth from penn-action-dataset
        gt_path = args.gt
        gt_data = scipy.io.loadmat(gt_path)
        target_bboxes = gt_data['bbox']

    if args.trial == 'True':
        trial = True
    elif args.trial == 'False':
        trial = False
    else:
        raise ValueError('Trial can be only True or False. Use --trial "True" for 40 frames trial.')

    if args.target_keypoints == '':
        raise ValueError('target_keypoints must not be empty, use --target-keypoints path_to_target_keypoints.json')
    else:
        target_keypoints = load_keypoints_from_json(args.target_keypoints)

    assert args.show or (args.output_root != '')
    assert args.input != ''

    output_file = None
    output_file_clean = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)

        filename = os.path.basename(args.input)
        base_name, ext = os.path.splitext(filename)

        output_file = os.path.join(args.output_root,
                                   f"{base_name}_adv{ext}")
        output_file_clean = os.path.join(args.output_root,
                                   f"{base_name}_adv_clean{ext}")
        if args.input == 'webcam':
            output_file += f'_adv{ext}'
            output_file_clean += f'_adv_clean{ext}'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_adv_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':

        # inference
        pred_instances = process_one_image(args, args.input,
                                           pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)

        video_writer = None
        video_writer_clean = None
        pred_instances_list = []
        frame_idx = 0

        all_keypoints = []

        # trial_limit for tests
        trial_limit = 25

        # total frames for console progress logging
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if trial and total_frames > trial_limit:
            print(f"\n===== RUNNING TRIAL VIDEO ({total_frames} frames) =====")
            total_frames = trial_limit
        else:
            print(f"\n===== RUNNING FULL VIDEO ({total_frames} frames) =====")
        
        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break
            
            if trial and frame_idx > trial_limit:
                print(f"\nTrial has ended after {trial_limit} frames.")
                break
            
            current_gt_bbox = target_bboxes[frame_idx-1]

            # displaying progress
            print("="*30)
            print(f"[{frame_idx}/{total_frames}] processing frame")

            detector=None

            # ========== ATTACK ==========
            if frame_idx - 1 < len(target_keypoints):
                frame, P_prev = apply_attack(
                    frame,
                    target_keypoints[frame_idx - 1],
                    args,
                    detector=detector,
                    pose_estimator=pose_estimator,

                    # ATTACK HYPERPARAMETERS 
                    # =============================================================
                    N = ATTACK_PARAMS['N'],                        # default 5
                    K = ATTACK_PARAMS['K'],                        # default 3
                    eps = ATTACK_PARAMS['eps'],                    # default 1.0
                    delta = ATTACK_PARAMS['delta'],                # default 0.1
                    # weight = ATTACK_PARAMS['weight'],            # default 0.9
                    # decay = ATTACK_PARAMS['decay'],              # default 0.9
                    perturb_max = ATTACK_PARAMS['perturb_max'],    # default 5.0
                    # para_rate = ATTACK_PARAMS['para_rate'],      # default 0.9
                    # noise_scale=ATTACK_PARAMS['noise_scale'],    # default 20.0
                    # =============================================================


                    prev_perturbation=P_prev if 'P_prev' in locals() else None,
                    gt_bbox=current_gt_bbox,
                    SCORE=OKS_score
                )
            else:
                P_prev = np.zeros_like(frame, dtype=np.float32)

            # copy of the adversarial frame for saving the video without keypoints
            frame_to_visualize = frame.copy()

            # pose estimation on the adversarial frame
            pred_instances = process_one_image(args, frame, detector,
                                               pose_estimator, visualizer,
                                               0.001, gt_bbox=current_gt_bbox)

            frame_keypoints = []
            if pred_instances is not None:
                for inst in split_instances(pred_instances):
                    frame_keypoints.append(inst['keypoints'])
            all_keypoints.append(frame_keypoints)

            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                # video with keypoints
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        30,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))
                
                # adversarial video
                if video_writer_clean is None:
                    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer_clean = cv2.VideoWriter(
                        output_file_clean,
                        fourcc,
                        30,  # saved fps
                        (frame_to_visualize.shape[1], frame_to_visualize.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))
                video_writer_clean.write(frame_to_visualize)

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        if video_writer_clean:
            video_writer_clean.release()

        cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the KEYPOINTS output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)

        print_log(
            f'the CLEAN output {input_type} has been saved at {output_file_clean}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
