from perturbations import orthogonal_perturbation, forward_perturbation, get_diff
from tools import OKS_score, IoU_score
import numpy as np

import mmcv
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


prev_target_keypoints = None # variable storing the previous frame

def process_one_image(args,
                      img,
                      detector=None,
                      pose_estimator=None,
                      visualizer=None,
                      show_interval=0,
                      gt_bbox=None):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    if detector is not None:
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
                iou_score = IoU_score(bboxes[i], gt_bbox)
                iou_bboxes.append(iou_score)

            best_idx = np.argmax(iou_bboxes)

            # MMPose requires bboxes as a 2D numpy array, e.g., shape (1, 4)
            bboxes_new = []
            threshold = 0.25 # threshold for bbox accept

            if iou_bboxes[best_idx] >= threshold:
                bboxes_new.append(bboxes[best_idx])
                bboxes = np.array(bboxes_new)
            else:
                bboxes = np.empty((0, 4), dtype=np.float32)
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)

    else:
        if gt_bbox is None or np.size(gt_bbox) == 0:
            print("Missing gt_bbox. Skipping keypoint detection for this frame.")
            
            if isinstance(img, str):
                img = mmcv.imread(img, channel_order='rgb')
            elif isinstance(img, np.ndarray):
                img = mmcv.bgr2rgb(img)

            if visualizer is not None:
                visualizer.set_image(img)

            return None
    
        bboxes = np.array([gt_bbox[:4]], dtype=np.float32)

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
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def apply_attack(frame, target_keypoints, args, detector=None, pose_estimator=None, 
                 prev_perturbation=None, eps=1.0, delta=0.1, N=5, K=3,
                 weight=0.9, para_rate=0.9, perturb_max=5.0, decay=0.8,
                 noise_scale=20.0, gt_bbox=None, SCORE=OKS_score):
    """
    Hyperparameters:
        eps: Step size in the forward perturbation (larger value = stronger effect).
        delta: Scale of the orthogonal perturbation (larger value = stronger effect).
        N: Number of sampled candidates.
        K: Number of iterations per frame.
        weight: How much the clean sample retains the previous perturbation
        para_rate: Score importance weight in the [0, 1] range (lower = temporal features are more important, higher = spatial features are more important).
        perturb_max: RMSE-norm limit (maximum allowed difference between images).
        decay: Retention rate of the previous perturbation (larger value = retains more of the previous noise).
        noise_scale: Strength of the initial noise.
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

    H, W = frame.shape[:2]

    # scaling epsilon to the image size
    D_sqrt = np.sqrt(frame.size)
    eps = eps * D_sqrt

    # baseline predictions and score for the original frame
    pred_orig = process_one_image(args=args, img=frame, detector=detector, pose_estimator=pose_estimator, visualizer=None, gt_bbox=gt_bbox)

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

    # initial canvas: contains the previous perturbation
    clean_sample = clean_sample_init + weight * prev_perturbation
    
    # noise_sample as a noisy version of clean_sample_init
    noise = np.random.normal(scale=noise_scale, size=clean_sample_init.shape).astype(np.float32)
    noise_sample = np.clip(clean_sample_init + noise, IMG_MIN, IMG_MAX)

    # initial adversarial_sample
    adversarial_sample = clean_sample.copy()
    
    # small step initializing the direction of change (forward step towards noise_sample)
    trial = adversarial_sample + forward_perturbation(eps, clean_sample, noise_sample)
    trial = np.clip(trial, IMG_MIN, IMG_MAX)
    pred_trial = process_one_image(args=args, img=np.clip(trial + 128.0, 0, 255).astype(np.uint8), detector=detector, pose_estimator=pose_estimator, visualizer=None, gt_bbox=gt_bbox)

    trial_score_spatial = SCORE(pred_trial, target_keypoints, area=area)
    if (prev_target_keypoints is None):
        trial_score_temporal = trial_score_spatial
    else:
        trial_score_temporal = SCORE(pred_trial, prev_target_keypoints, area=area)
    trial_score = para_rate * trial_score_spatial + (1 - para_rate) * trial_score_temporal
    trial_score = float(np.asarray(trial_score).item())
    
    threshold = para_rate * orig_score + (1 - para_rate) * trial_score
    adversarial_sample = trial.copy()

    best_adversarial_score = float(trial_score)
    best_adversarial_sample = adversarial_sample.copy() 
    diff_init = trial - clean_sample_init
    best_adversarial_rmse = np.linalg.norm(diff_init) / np.sqrt(diff_init.size)
    print(f"  [INITIAL] Start | score: {best_adversarial_score:.4f} | rmse: {best_adversarial_rmse:.6f}")

    # main loop:
    for k in range(K):
        # ============== ORTHOGONAL PERTURBATION ==============
        # N candidates generated orthogonally to the direction (noise_sample - adversarial)
        candidates = []
        for i in range(N):
            cand = adversarial_sample + orthogonal_perturbation(delta, noise_sample, adversarial_sample)

            # ================================= [Master's Thesis Variant] ================================= 
            # cand = adversarial_sample + orthogonal_perturbation(delta, clean_sample, adversarial_sample)
            # =============================================================================================

            diff = cand - clean_sample_init
            diff = np.clip(diff, -perturb_max, perturb_max)
            cand = np.clip(clean_sample_init + diff, IMG_MIN, IMG_MAX)

            pred_cand = process_one_image(args=args, img=np.clip(cand + 128.0, 0, 255).astype(np.uint8), detector=detector, pose_estimator=pose_estimator, visualizer=None, gt_bbox=gt_bbox)
            s_spatial = SCORE(pred_cand, target_keypoints, area=area)
            if (prev_target_keypoints is None):
                s_temporal = s_spatial
            else:
                s_temporal = SCORE(pred_cand, prev_target_keypoints, area=area)
            s = para_rate * s_spatial + (1 - para_rate) * s_temporal
            s = float(np.asarray(s).item())
            candidates.append((s, cand))
            print(f"        [ORTHOGONAL] cand {i+1} | score: {s:.4f} | delta: {delta:.4f}")

        s_vals = np.array([float(np.asarray(c[0]).item()) for c in candidates], dtype=float)
        best_idx = int(np.argmin(s_vals)) # lower score is considered the best
        best_s = float(s_vals[best_idx]) # best score
        best_cand = candidates[best_idx][1] # best candidate

        if best_s >= threshold: # if the best candidate score is higher (worse) than the threshold, decrease delta
            delta *= 0.9

        print(f'          [BEST CAND] best_score={best_s:.4f}')

        # ============== FORWARD PERTURBATION ==============
        # forward refinement (using noise_sample as target_sample)
        rmse = 0.0 # image difference: close to 0 means unchanged, high means heavily distorted
        current_eps = eps
        for i in range(N):
            if current_eps < 0.5:
                print(f"          [BREAK] epsilon too low: {current_eps:.4f}")
                break

            cand2 = best_cand + forward_perturbation(current_eps, best_cand, noise_sample)

            # ========================== [Master's Thesis Variant] ========================== 
            # cand2 = best_cand + forward_perturbation(current_eps, best_cand, clean_sample)
            # ===============================================================================

            diff2 = cand2 - clean_sample_init
            diff2 = np.clip(diff2, -perturb_max, perturb_max)
            cand2 = np.clip(clean_sample_init + diff2, IMG_MIN, IMG_MAX)

            pred_cand2 = process_one_image(args=args, img=np.clip(cand2 + 128.0, 0, 255).astype(np.uint8), detector=detector, pose_estimator=pose_estimator, visualizer=None, gt_bbox=gt_bbox)
            s2_spatial = SCORE(pred_cand2, target_keypoints, area=area)
            if (prev_target_keypoints is None):
                s2_temporal = s2_spatial
            else:
                s2_temporal = SCORE(pred_cand2, prev_target_keypoints, area=area)
            s2= para_rate * s2_spatial + (1 - para_rate) * s2_temporal
            s2 = float(np.asarray(s2).item())
            diff = cand2 - clean_sample_init
            # mean deviation per pixel (RMSE)
            rmse = np.linalg.norm(diff) / np.sqrt(diff.size)

            # if the new candidate is lower (better) than the current one, update
            if s2 <= threshold:
                adversarial_sample = cand2.copy()
                threshold = s2
                if s2 < best_adversarial_score:
                    best_adversarial_score = float(s2)
                    best_adversarial_sample = adversarial_sample.copy()
                    best_adversarial_rmse = rmse
                current_eps *= 1.2
                print(f"        [FORWARD SUCCES] cand {i+1} | score: {s2:.4f} | rmse {rmse:.6f} | eps {current_eps:.4f}")
            else:
                current_eps *= 0.8
                print(f"        [FORWARD FAILED] cand {i+1} | score: {s2:.4f} | rmse {rmse:.6f} | eps {current_eps:.4f}")

        print(f'  [PERTURB] Iter {k+1}/{K}: best_score={best_adversarial_score:.4f}, threshold={threshold:.4f}, rmse={best_adversarial_rmse:.6f}')

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