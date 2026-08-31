import subprocess
import os
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)

my_env = os.environ.copy()

if "PYTHONPATH" in my_env:
    my_env["PYTHONPATH"] = f"{ROOT_DIR}{os.pathsep}{my_env['PYTHONPATH']}"
else:
    my_env["PYTHONPATH"] = ROOT_DIR


# Helper function for printing to the console and logging
def log_print(text, file):
    print(text)
    print(text, file=file)
    file.flush()


# STEP 1 - CLEAN PREDICTION
def clean_pred(name, folder_exp_name, detector_cfg, detector_ckpt, pose_est_cfg,
             pose_est_ckpt, input_path, mat_kp_path_gt, trial, experiments_folder, log_path):
    with open(log_path, "a", encoding="utf-8") as log_file:

        # script path
        script = "../demo/topdown_demo_with_mmdet_clean.py"

        # output folder (for saving results) with an 'og' subfolder for original detection results
        output_folder = f"../{experiments_folder}/{folder_exp_name}/og"

        log_print(f"\n >>> STEP 1 - CLEAN PREDICTION", log_file)
        log_print(f"Running script {script} for video {name}...", log_file)
        log_print(f"="*80+"\n", log_file)

        # script execution
        command = [
            "python",
            script,
            detector_cfg,
            detector_ckpt,
            pose_est_cfg,
            pose_est_ckpt,
            "--input", input_path,
            "--output-root", output_folder,
            "--save-predictions",
            "--trial", trial,
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
# INDIVIDUAL FOR EVERY EXPERIMENT


# STEP 3 - VISUALIZING DIFFERENCES
def visualization_save(input_path, folder_exp_name, name, experiments_folder, log_path):
    with open(log_path, "a", encoding="utf-8") as log_file:

        # script path
        test_script_1 = "../oks_tools/visualization.py" 

        # path to the original video for result visualization
        video_path = input_path
        # path to the keypoints BEFORE the attack
        org_kpts_path = f"../{experiments_folder}/{folder_exp_name}/og/results_{name}.json"
        # path to the keypoints AFTER the attack
        adv_kpts_path = f"../{experiments_folder}/{folder_exp_name}/adv/results_adv_{name}.json"
        # path to the output folder where the test video will be saved
        output_test_path = f"../{experiments_folder}/{folder_exp_name}/results"
        # thickness of points and lines for the drawn keypoints
        point = 5
        line = 1

        log_print(f"\n >>> STEP 3 - VISUALIZING DIFFERENCES", log_file)
        log_print(f"Running script {test_script_1} for video {name}...", log_file)
        log_print(f"="*80+"\n", log_file)

        # script execution
        command = [
            "python",
            test_script_1,
            "--video", video_path,
            "--org_keypoints", org_kpts_path,
            "--adv_keypoints", adv_kpts_path,
            "--output_path", output_test_path,
            "--point", str(point),
            "--line", str(line)
        ]
        
        result = subprocess.run(
            command, 
            env=my_env, 
            stdout=log_file, 
            stderr=subprocess.STDOUT, 
            text=True
        )

        if result.returncode != 0:
            log_print(f"Error in script {test_script_1}", log_file)


# STEP 4 - ACCURACY
def accuracy_measure(mat_kp_path_gt, folder_exp_name, name, model_name, experiments_folder, log_path):
    with open(log_path, "a", encoding="utf-8") as log_file:

        # script path
        test_script_2 = '../oks_tools/measures.py'

        # paths to the JSON files containing keypoints
        gt_kp = mat_kp_path_gt       # ground truth
        org_kp = f'../{experiments_folder}/{folder_exp_name}/og/results_{name}.json'       # original results
        adv_kp = f'../{experiments_folder}/{folder_exp_name}/adv/results_adv_{name}.json'  # adversarial results

        output_folder_path = f'../{experiments_folder}/{folder_exp_name}/results'

        log_print(f"\n >>> STEP 4 - ACCURACY", log_file)
        log_print(f"Running script {test_script_2} for video {name}...", log_file)
        log_print(f"="*80+"\n", log_file)

        # script execution
        command = [
            "python",
            test_script_2,
            "--ground", gt_kp,
            "--test_org", org_kp,
            "--test_adv", adv_kp,
            "--output", output_folder_path,
            "--model", model_name,
            "--test", folder_exp_name
        ]
        
        result = subprocess.run(
            command, 
            env=my_env, 
            stdout=log_file, 
            stderr=subprocess.STDOUT, 
            text=True
        )

        if result.returncode != 0:
            log_print(f"Error in script {test_script_2}", log_file)


# STEP 5 - ACTION RECOGNITION
def action_recognition(experiments_folder, folder_exp_name, gt, log_path):
    with open(log_path, "a", encoding="utf-8") as log_file:

        # script path
        script = '../oks_tools/action_recognition.py'

        # paths to the folders containing JSON files
        og_kpts_path = f"../{experiments_folder}/{folder_exp_name}/og"
        adv_kpts_path = f"../{experiments_folder}/{folder_exp_name}/adv"

        output_folder = f"../{experiments_folder}/{folder_exp_name}/results"

        log_print(f"\n >>> STEP 5 - ACTION RECOGNITION", log_file)
        log_print(f"Running script {script}...", log_file)
        log_print(f"="*80+"\n", log_file)

        # script execution
        command = [
                "python",
                script,
                "--exp", folder_exp_name,
                "--og-kpts", og_kpts_path,
                "--adv-kpts", adv_kpts_path,
                "--output", output_folder,
                "--gt", gt
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