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
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

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

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0,
                      gt_bbox=None):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # ======== FILTERING OUT EXTRA PERSON DETECTIONS ========
    iou_bboxes = []
    if gt_bbox is not None and len(bboxes) > 0:
        for i in range(0, len(bboxes)):
            x_left = np.maximum(bboxes[i][0], gt_bbox[0])
            y_top = np.maximum(bboxes[i][1], gt_bbox[1])
            x_right = np.minimum(bboxes[i][2], gt_bbox[2])
            y_bot = np.minimum(bboxes[i][3], gt_bbox[3])
            
            if x_right < x_left or y_bot < y_top:
                iou_bboxes.append(0.0)
                continue

            intersection_area = (x_right - x_left) * (y_bot - y_top)

            bbox_det = (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])
            bbox_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

            union_area = bbox_det + bbox_gt - intersection_area

            if union_area == 0:
                iou_bboxes.append(0.0)
                continue

            iou_score = intersection_area / union_area

            iou_bboxes.append(iou_score)

        best_idx = np.argmax(iou_bboxes)

        bboxes_new = []
        threshold = 0.25 # threshold for bbox accept

        if iou_bboxes[best_idx] >= threshold:
            bboxes_new.append(bboxes[best_idx])
            bboxes = np.array(bboxes_new)
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
    else:
            bboxes = np.empty((0, 4), dtype=np.float32)
        

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # ======== FILTERING BY KEYPOINT VISIBILITY THRESHOLD ========
    pred_instances = data_samples.get('pred_instances', None)

    if pred_instances is not None:
        scores = pred_instances.keypoint_scores
        mean_scores = np.mean(scores, axis=1)
        threshold = 0.5 # CLASSIFICATION THRESHOLD FOR KEYPOINT VISIBILITY
        valid_mask = mean_scores > threshold
        pred_instances = pred_instances[valid_mask]

        data_samples.pred_instances = pred_instances

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr
        )

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
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
        '--trial',
        type=str,
        default='False',
        help='Start attack full or trial')
    parser.add_argument(
        '--gt',
        type=str,
        default=None,
        help='Mat file with groundtruth of video')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # load .mat file with groundtruth from penn-action-dataset
    gt_path = args.gt
    gt_data = scipy.io.loadmat(gt_path)
    bboxes_gt = gt_data['bbox']

    if args.trial == 'True':
        trial = True
    elif args.trial == 'False':
        trial = False
    else:
        raise ValueError('Trial can be only True or False. Use --trial "True" for 40 frames trial.')

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

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
        pred_instances = process_one_image(args, args.input, detector,
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
        pred_instances_list = []
        frame_idx = 0

        all_keypoints = []

        # trial_limit for testing
        trial_limit = 25

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if trial and total_frames > trial_limit:
            print(f"\n===== RUNNING TRIAL VIDEO ({trial_limit} frames) =====")
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
            
            current_gt_bbox = bboxes_gt[frame_idx-1]

            # topdown pose estimation
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

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        30,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

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
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)

    
if __name__ == '__main__':
    main()
