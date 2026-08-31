import os
import subprocess
import sys

from pathlib import Path
from argparse import ArgumentParser

# random seed
tools_dir = "../oks_tools"
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from utils_seed import set_deterministic_seed
from experiment_core import log_print, clean_pred, visualization_save, accuracy_measure, action_recognition

set_deterministic_seed(42)


ROOT_DIR = str(Path(__file__).resolve().parent.parent)

my_env = os.environ.copy()

if "PYTHONPATH" in my_env:
    my_env["PYTHONPATH"] = f"{ROOT_DIR}{os.pathsep}{my_env['PYTHONPATH']}"
else:
    my_env["PYTHONPATH"] = ROOT_DIR


# STEP 0 - TEST INITIALIZATION
def run_experiment(exp_id, experiments_folder, pose_est_cfg, pose_est_ckpt):

    # path to the input file we want to analyze
    input_path = f"../penn-action-dataset/videos/{exp_id}-vis.mkv"
    # original file name, which will be reused in subsequent steps 
    name = os.path.splitext(os.path.basename(input_path))[0]
    # experiment folder name (to save tests on the same video in separate directories)
    folder_exp_name = name+"_exp"

    # detector settings
    detector_cfg = "../demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
    detector_ckpt = "../checkpoints/faster_rcnn_r50.pth"

    # GROUND TRUTH from PENN-ACTION-DATASET
    mat_kp_path_gt = f"../penn-action-dataset/labels/{exp_id}.mat"

    # Is it Trial (True = trial 25 frames)
    trial = 'False'

    # name of the model used for pose estimation and attack
    model_name = os.path.splitext(os.path.basename(pose_est_cfg))[0]

    # file with logs
    log_path = f"../{experiments_folder}/{folder_exp_name}/logs_{folder_exp_name}.txt"

    # create the directory if it does not exist
    log_path_dir = Path(log_path)
    log_path_dir.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_print(f"Experiment CLEAN_BBOX_NO_DET", log_file)
        log_print(f"Analyzed video: {name}", log_file)
        log_print(f"Model used: {model_name}", log_file) 


    # ================================================ full pipeline ================================================
    # 1 - clean prediction
    clean_pred(name=name, folder_exp_name=folder_exp_name, detector_cfg=detector_cfg, detector_ckpt=detector_ckpt, 
             pose_est_cfg=pose_est_cfg, pose_est_ckpt=pose_est_ckpt, input_path=input_path, mat_kp_path_gt=mat_kp_path_gt, 
             trial=trial, experiments_folder=experiments_folder, log_path=log_path)
    
    # 2 - adversarial attack
    oks_attack(name=name, folder_exp_name=folder_exp_name, 
             pose_est_cfg=pose_est_cfg, pose_est_ckpt=pose_est_ckpt, input_path=input_path, 
             trial=trial, experiments_folder=experiments_folder, log_path=log_path)

    # 3 - visualizing differences
    visualization_save(input_path=input_path, folder_exp_name=folder_exp_name, name=name, experiments_folder=experiments_folder, 
                     log_path=log_path)

    # 4 - accuracy
    accuracy_measure(mat_kp_path_gt=mat_kp_path_gt, folder_exp_name=folder_exp_name, name=name, model_name=model_name, 
                     experiments_folder=experiments_folder, log_path=log_path)

    # 5 - action recognition
    action_recognition(experiments_folder=experiments_folder, folder_exp_name=folder_exp_name, gt=mat_kp_path_gt, log_path=log_path)
    # ===============================================================================================================

    if os.path.exists(log_path):
        print(f"\nLogs saved to: {os.path.abspath(log_path)}")
    else:
        print(f"\nFailed to create log file for video {name}")


# STEP 2 - ADVERSARIAL ATTACK
def oks_attack(name, folder_exp_name, pose_est_cfg,
             pose_est_ckpt, input_path, trial, experiments_folder, log_path):
    with open(log_path, "a", encoding="utf-8") as log_file:

        # script path
        script = "../attacks/OKS_attack_no_det.py"

        # output folder (for saving results) with an 'adv' subfolder for adversarial detection results
        output_folder = f"../{experiments_folder}/{folder_exp_name}/adv"

        # target keypoints (keypoints saved in .json file)
        target_kpt = f"../{experiments_folder}/{folder_exp_name}/og/results_{name}.json"

        log_print(f"\n >>> STEP 2 - ADVERSARIAL ATTACK (without detector - ablation study)", log_file)
        log_print(f"Running script {script} for video {name}...", log_file)
        log_print(f"="*80+"\n", log_file)

        # script execution
        command = [
            "python",
            script,
            pose_est_cfg,
            pose_est_ckpt,
            "--input", input_path,
            "--output-root", output_folder,
            "--save-predictions",
            "--target-keypoints", target_kpt,
            "--trial", trial
        ]
        
        result = subprocess.run(
            command, 
            env=my_env, 
            stdout=log_file, 
            stderr=subprocess.STDOUT, 
            text=True
        )

        if result.returncode != 0:
            log_print(f"Error in script {script}", log_file)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input', 
        type=str, 
        default='', 
        help='Path to file with selected files')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Path to experiment output folder')
    parser.add_argument(
        '--model',
        type=str,
        default='res50',
        help='Name of model to test')
    
    args = parser.parse_args()

    if args.input == '':
        raise ValueError('Input must not be empty, use --input path_to_selected_files_file.txt')
    else:
        selected_files = args.input
    
    if args.output == '':
        raise ValueError('Output must not be empty, use --output path_to_experiments_output_folder')
    else:
        experiments_folder = args.output

    model_name = args.model.lower() # default ResNet50 (res50)

    # pose estimator settings
    if model_name == 'res50':
        pose_est_cfg = f"../configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py"   # <--- config
        pose_est_ckpt = f"../checkpoints/res50.pth"        # <--- checkpoint
    elif model_name == 'mobilenetv2':
        pose_est_cfg = f"../configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_mobilenetv2_8xb64-210e_coco-256x192.py"   # <--- config
        pose_est_ckpt = f"../checkpoints/mobilenetv2.pth"        # <--- checkpoint
    else:
        raise ValueError("Available models: [ res50 / mobilenetv2 ], use --model X")

    errors_counter = 0

    with open(selected_files, 'r') as f:
        for line in f:
            experiment_id = line.strip()

            if not experiment_id:
                continue
            
            if not os.path.exists(f'../penn-action-dataset/videos/{experiment_id}-vis.mkv'):
                print(f"WARNING: Video {experiment_id} does not exist. Skipping...")
                errors_counter += 1
                continue

            run_experiment(experiment_id, experiments_folder, pose_est_cfg, pose_est_ckpt)

    if errors_counter > 0:
        print(f"Failed to process {errors_counter} videos.")
    else:
        print(f"Successfully processed all videos.")

if __name__ == '__main__':
    main()