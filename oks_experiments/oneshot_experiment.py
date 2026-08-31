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
from experiment_core import log_print, visualization_save, accuracy_measure, action_recognition

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

    # GROUND TRUTH from PENN-ACTION-DATASET
    mat_kp_path_gt = f"../penn-action-dataset/labels/{exp_id}.mat"

    # Is it Trial (True = trial 25 frames)
    trial = 'False'

    # name of the model used for pose estimation and attack
    model_name = os.path.splitext(os.path.basename(pose_est_ckpt))[0]

    # file with logs
    log_path = f"../{experiments_folder}/{folder_exp_name}/logs_{folder_exp_name}.txt"

    # create the directory if it does not exist
    log_path_dir = Path(log_path)
    log_path_dir.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_print(f"Experiment YOLO_POSE", log_file)
        log_print(f"Analyzed video: {name}", log_file)
        log_print(f"Model used: {model_name}", log_file) 

    
    # ================================================ full pipeline ================================================
    # 1 - clean prediction
    clean_pred(name=name, folder_exp_name=folder_exp_name, pose_est_cfg=pose_est_cfg, pose_est_ckpt=pose_est_ckpt, 
             input_path=input_path, experiments_folder=experiments_folder, log_path=log_path, mat_kp_path_gt=mat_kp_path_gt)
    
    # 2 - adversarial attack
    oks_attack(name=name, folder_exp_name=folder_exp_name, pose_est_cfg=pose_est_cfg, pose_est_ckpt=pose_est_ckpt, 
               input_path=input_path, experiments_folder=experiments_folder, log_path=log_path, mat_kp_path_gt=mat_kp_path_gt)

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

# STEP 1 - CLEAN PREDICTION
def clean_pred(name, folder_exp_name, pose_est_cfg, pose_est_ckpt, input_path, 
             experiments_folder, log_path, mat_kp_path_gt):
    with open(log_path, "a", encoding="utf-8") as log_file:
        
        # script path
        script = "../demo/bottomup_demo_clean.py"

        # output folder (for saving results) with an 'og' subfolder for original detection results
        output_folder = f"../{experiments_folder}/{folder_exp_name}/og"

        log_print(f"\n >>> STEP 1 - CLEAN PREDICTION", log_file)
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
            "--gt", mat_kp_path_gt
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



# STEP 2 - ADVERSARIAL ATTACK
def oks_attack(name, folder_exp_name, pose_est_cfg, pose_est_ckpt, 
               input_path, experiments_folder, log_path, mat_kp_path_gt):
    with open(log_path, "a", encoding="utf-8") as log_file:

        # script path
        script = "../attacks/OKS_attack_oneshot.py"

        # output folder (for saving results) with an 'adv' subfolder for adversarial detection results
        output_folder = f"../{experiments_folder}/{folder_exp_name}/adv"

        # target keypoints (keypoints saved in .json file)
        target_kpt = f"../{experiments_folder}/{folder_exp_name}/og/results_{name}.json"

        log_print(f"\n >>> STEP 2 - ADVERSARIAL ATTACK", log_file)
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
            "--gt", mat_kp_path_gt
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
        default='s',
        help='YOLO-Pose model')
    
    args = parser.parse_args()

    if args.input == '':
        raise ValueError('Input must not be empty, use --input path_to_selected_files_file.txt')
    else:
        selected_files = args.input
    
    if args.output == '':
        raise ValueError('Output must not be empty, use --output path_to_experiments_output_folder')
    else:
        experiments_folder = args.output

    model = args.model.lower()

    # pose estimator settings
    if model == 's':
        pose_est_cfg = f"../configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_s_8xb32-300e_coco-640.py"   # <--- config
        pose_est_ckpt = f"../checkpoints/yoloxpose_s.pth" 
    elif model == 'm':
        pose_est_cfg = f"../configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_m_8xb32-300e_coco-640.py"   # <--- config
        pose_est_ckpt = f"../checkpoints/yoloxpose_m.pth" 
    else:
        raise ValueError("Available YOLO-Pose models: [ Tiny / S / M ], use --model X")

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