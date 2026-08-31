# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_bottomup, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import split_instances

import scipy.io
import sys

from tools import load_keypoints_from_json, OKS_score, IoU_score
from attack_config import ATTACK_PARAMS
from perturbations import orthogonal_perturbation, forward_perturbation, get_diff

# random seed
tools_dir = "../oks_tools"
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from utils_seed import set_deterministic_seed

set_deterministic_seed(42)

prev_target_keypoints = None # variable storing the previous frame

def process_one_image(args,
                      img,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0,
                      gt_bbox=None):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # inference a single image
    batch_results = inference_bottomup(pose_estimator, img)
    results = batch_results[0]

    # ======== FILTERING OUT EXTRA PERSON DETECTIONS ========
    if gt_bbox is not None and len(results.pred_instances) > 0:
        pred_bboxes = results.pred_instances.bboxes

        best_iou = -1.0
        best_idx = 0

        for idx, bbox in enumerate(pred_bboxes):
            current_iou = IoU_score(bbox, gt_bbox)

            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx

        iou_threshold = 0.25

        if best_iou >= iou_threshold:
            results.pred_instances = results.pred_instances[best_idx : best_idx + 1]
        else:
            empty_mask = np.zeros(len(results.pred_instances), dtype=bool)
            results.pred_instances = results.pred_instances[empty_mask]

    # ======== FILTERING BY KEYPOINT VISIBILITY THRESHOLD ========
    if len(results.pred_instances) > 0:
        scores = results.pred_instances.keypoint_scores
        mean_scores = np.mean(scores, axis=1)
        kpt_threshold = 0.5 # CLASSIFICATION THRESHOLD FOR KEYPOINT VISIBILITY
        valid_mask = mean_scores > kpt_threshold

        results.pred_instances = results.pred_instances[valid_mask]

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=results,
            draw_gt=False,
            draw_bbox=False,
            draw_heatmap=args.draw_heatmap,
            show_kpt_idx=args.show_kpt_idx,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    return results.pred_instances


def apply_attack(frame, target_keypoints, args, pose_estimator=None, 
                 prev_perturbation=None, eps=1.0, delta=0.1, N=5, K=3,
                 weight=0.9, para_rate=0.9, perturb_max=5.0, decay=0.8,
                 noise_scale=20.0, gt_bbox=None, SCORE=OKS_score):
    """
    Apply_attack atak IoU który używa:
      - forward_perturbation(epsilon, prev_sample, target_sample)
      - orthogonal_perturbation(delta, prev_sample, target_sample)

    Hiperparametry:
      - eps - rozmiar kroku w forward perturbation - im większe tym mocniejsze
      - delta - skala ortogonalnej perturbacji - im większe tym mocniejsze
      - N - ile kandydatów losuje
      - K - ile iteracji na klatke
      - weight - jak bardzo clean_sample zawiera prev_perturbation - im mniejsze tym szybciej adaptuje nowe
      - para_rate - waga ważności score przedział 0-1, im niższa tym temporal ważniejszy, im wyższa tym spatial ważniejszy
      - perturb_max - limit rmse-norm (różnicy między obrazami)
      - decay - jak bardzo zachowuje poprzednia perturbacje - im wieksze tym bardziej zachowuje poprzedni szum
      - noise_scale - moc wstępnego szumu
    """
    global prev_target_keypoints

    IMG_MIN = -128.0 
    IMG_MAX = 127.0

    # area calculated based on bounding boxes
    if gt_bbox is not None and len(gt_bbox) == 4:
        width = gt_bbox[2] - gt_bbox[0]
        height = gt_bbox[3] - gt_bbox[1]
        area = float(width * height)
    else:
        area = None

    if prev_perturbation is None:
        prev_perturbation = np.zeros_like(frame, dtype=np.float32)

    # baseline predictions and score for the original frame
    pred_orig = process_one_image(args=args, img=frame, pose_estimator=pose_estimator, visualizer=None, gt_bbox=gt_bbox)

    if pred_orig is None:
        return frame, prev_perturbation

    orig_score_spatial = SCORE(pred_orig, target_keypoints, area=area)
    if (prev_target_keypoints is None):
        orig_score_temporal = orig_score_spatial
    else:
        orig_score_temporal = SCORE(pred_orig, prev_target_keypoints, area=area)
    orig_score = para_rate * orig_score_spatial + (1 - para_rate) * orig_score_temporal

    frame_f = frame.astype(np.float32)
    # centering for stability: range -128 .. 127
    clean_sample_init = frame_f - 128.0

    best_adversarial_score = orig_score 
    best_adversarial_sample = clean_sample_init.copy()
    best_adversarial_rmse = 0.0

    print(f"  [INITIAL] Start | score: {best_adversarial_score:.4f} | rmse: {best_adversarial_rmse:.6f}")

    total_iterations = K * N

    # main loop: introducing random noise
    for i in range(total_iterations):
        # generating random noise
        noise = np.random.normal(scale=noise_scale, size=clean_sample_init.shape).astype(np.float32)

        # adding the previous perturbation
        total_noise = noise + (weight * prev_perturbation)

        # clipping to maintain imperceptibility
        total_noise = np.clip(total_noise, -perturb_max, perturb_max)

        # adding noise to the image
        cand = clean_sample_init + total_noise
        cand = np.clip(cand, IMG_MIN, IMG_MAX)

        # prediction on the candidate
        pred_cand = process_one_image(args=args, img=np.clip(cand + 128.0, 0, 255).astype(np.uint8), pose_estimator=pose_estimator, visualizer=None, gt_bbox=gt_bbox)

        # calculating OKS
        s_spatial = SCORE(pred_cand, target_keypoints, area=area) 
        if (prev_target_keypoints is None):
            s_temporal = s_spatial
        else:
            s_temporal = SCORE(pred_cand, prev_target_keypoints, area=area)
        s = para_rate * s_spatial + (1 - para_rate) * s_temporal
        s = float(np.asarray(s).item())

        # calculating RMSE for logging
        diff = cand - clean_sample_init
        rmse = np.linalg.norm(diff) / np.sqrt(diff.size)

        print(f"        [RANDOM SEARCH] cand {i+1}/{total_iterations} | score: {s:.4f} | rmse: {rmse:.6f}")

        if s < best_adversarial_score:
            best_adversarial_score = s
            best_adversarial_sample = cand.copy()
            best_adversarial_rmse = rmse

    print(f'  [PERTURB END]: best_score={best_adversarial_score:.4f}, rmse={best_adversarial_rmse:.6f}')

    # finalization
    final_adv = np.clip(best_adversarial_sample + 128.0, 0, 255).astype(np.uint8)
    current_noise = best_adversarial_sample - clean_sample_init

    # blending the old perturbation with the newly found one
    if np.max(np.abs(prev_perturbation)) == 0.0:
        new_perturbation = current_noise
    else:
        new_perturbation = decay * prev_perturbation + (1.0 - decay) * current_noise

    new_perturbation = np.clip(new_perturbation, IMG_MIN, IMG_MAX)

    # updating the previous frame (prev_target_keypoints)
    prev_target_keypoints = target_keypoints

    return final_adv, new_perturbation


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
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
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--target-keypoints',
        type=str,
        default='',
        help='root of the target keypoints JSON file. ')
    parser.add_argument(
        '--gt',
        type=str,
        default='',
        help='Mat file with groundtruth of video')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.show or (args.output_root != '')
    assert args.input != ''

    if args.target_keypoints == '':
        raise ValueError('target_keypoints must not be empty, use --target-keypoints path_to_target_keypoints.json')
    else:
        target_keypoints = load_keypoints_from_json(args.target_keypoints)

    # load .mat file with groundtruth from penn-action-dataset
    if args.gt == '':
        raise ValueError('groundtruth must not be empty, use --gt path_to_gt_folder.json')
    else:
        gt_path = args.gt
        gt_data = scipy.io.loadmat(gt_path)
        target_bboxes = gt_data['bbox']

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

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # build visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':
        # inference
        pred_instances = process_one_image(
            args, args.input, model, visualizer, show_interval=0)

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

        # total frames for console progress logging
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n===== RUNNING FULL VIDEO ({total_frames} frames) =====")

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break
            
            current_gt_bbox = target_bboxes[frame_idx-1]

            # displaying progress
            print("="*30)
            print(f"[{frame_idx}/{total_frames}] processing frame")

            # ========== ATTACK ==========
            if frame_idx - 1 < len(target_keypoints):
                frame, P_prev = apply_attack(
                    frame,
                    target_keypoints[frame_idx - 1],
                    args,
                    model,

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
                    SCORE=OKS_score # score calculated during the attack
                )
            else:
                P_prev = np.zeros_like(frame, dtype=np.float32)

            # copy of the adversarial frame for saving the video without keypoints
            frame_to_visualize = frame.copy()

            # pose estimation on the adversarial frame
            pred_instances = process_one_image(args, frame,
                                               model, visualizer,
                                               0.001, gt_bbox=current_gt_bbox)

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
                    meta_info=model.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
