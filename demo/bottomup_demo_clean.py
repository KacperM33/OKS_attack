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
        '--gt',
        type=str,
        default=None,
        help='Mat file with groundtruth of video')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.show or (args.output_root != '')
    assert args.input != ''

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mkv'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    if args.gt:
        # load .mat file with groundtruth from penn-action-dataset
        gt_path = args.gt
        gt_data = scipy.io.loadmat(gt_path)
        bboxes_gt = gt_data['bbox']
    else:
        raise ValueError('groundtruth must not be empty. Use --gt path_to_gt')


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
        pred_instances_list = []
        frame_idx = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n===== RUNNING FULL VIDEO ({total_frames} frames) =====")

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break
            
            current_gt_bbox = bboxes_gt[frame_idx-1]

            pred_instances = process_one_image(args, frame, model, visualizer,
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
