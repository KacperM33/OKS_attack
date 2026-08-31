# *A Black-Box Adversarial Attack on Human Pose Estimation and Keypoint-Based Action Recognition Models*

> 🇵🇱 **Polska wersja językowa / Dokumentacja pracy magisterskiej:**  
> Instrukcję odtworzenia eksperymentów specyficznych dla pracy magisterskiej znajdziesz w [README_PL_THESIS.md](README_PL_THESIS.md).

Proposed **OKS Attack**, a decision-based black-box attack that uses Object Keypoint Similarity (OKS) as the attack feedback signal, directly targeting the spatial structure of human poses.
Experiments on the Penn Action dataset show that **OKS Attack** consistently reduces pose quality across evaluated pose estimators. In a downstream crossdataset action-recognition evaluation, the attack reduces accuracy and outperforms query-matched randomnoise perturbations. The attack is effective across both top-down and single-stage pose estimation models.

<!-- LINK -->


## TABLE OF CONTENTS

🔹 [📊 MAIN RESULTS](#main-results)
<br>🔹 [📁 CODE STRUCTURE](#code-structure) 
<br>🔹 [⚙️ ENVIRONMENT SETUP](#environment-setup) 
<br>🔹 [🗃️ DATASET](#dataset) 
<br>🔹 [🧠 PRE-TRAINED MODELS (CHECKPOINTS)](#pre-trained-models-checkpoints) 
<br>🔹 [📓 STEP-BY-STEP TUTORIAL](#tutorial) 
<br>🔹 [🧪 REPRODUCING EXPERIMENTS](#reproducing-experiments) 
<br>🔹 [✏️ CITATION](#citation) 
<br>🔹 [🌐 ACKNOWLEDGEMENTS](#acknowledgements) 

<!-- VVV Przed reproducing VVV -->
<!--  <br>🔹 [ATTACKS](#attacks)  -->



<a id="main-results"></a>
## 📊 MAIN RESULTS

**Figure 1.** Overview of the proposed attack.
![alt text](fig3_5.png)

**Figure. 2.** Examples of action prediction changes. Each row shows frames sampled from one video at a fixed interval of ∆ = 10 frames. Clean and adversarial pose predictions are overlaid in green and red; the absence of a skeleton indicates that no pose was detected. Boxes report clean and adversarial predictions with confidence scores.

![alt text](fig2.png)

**Table 1.** Mean OKS for clean and adversarial pose predictions. Results are reported for all videos, action-changed videos, and action-unchanged videos. Random noise denotes a query-matched baseline with the same perturbation budget as OKS Attack.

<center>
<table>
  <thead>
    <tr>
      <th rowspan="3">2D HPE</th>
      <th colspan="9"><center>Mean Object Keypoint Similarity</center></th>
    </tr>
    <tr>
      <th colspan="3"><center>All videos</center></th>
      <th colspan="3"><center>Action changed</center></th>
      <th colspan="3"><center>Action unchanged</center></th>
    </tr>
    <tr>
      <th>Orig</th>
      <th>OKS Attack</th>
      <th>Rand. noise</th>
      <th>Orig</th>
      <th>OKS Attack</th>
      <th>Rand. noise</th>
      <th>Orig</th>
      <th>OKS Attack</th>
      <th>Rand. noise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNet-50</td>
      <td>0.7361</td>
      <td><b>0.6045</b></td>
      <td>0.6924</td>
      <td>0.6189</td>
      <td><b>0.4200</b></td>
      <td>0.5078</td>
      <td>0.7633</td>
      <td><b>0.6472</b></td>
      <td>0.7154</td>
    </tr>
    <tr>
      <td>MobileNetV2</td>
      <td>0.6931</td>
      <td><b>0.5437</b></td>
      <td>0.6635</td>
      <td>0.4674</td>
      <td><b>0.3444</b></td>
      <td>0.4128</td>
      <td>0.7558</td>
      <td><b>0.6221</b></td>
      <td>0.6997</td>
    </tr>
    <tr>
      <td>YOLO-Pose M</td>
      <td>0.7547</td>
      <td><b>0.6745</b></td>
      <td>0.7121</td>
      <td>0.6592</td>
      <td><b>0.5362</b></td>
      <td>0.5911</td>
      <td>0.7857</td>
      <td><b>0.7194</b></td>
      <td>0.7418</td>
    </tr>
    <tr>
      <td>YOLO-Pose S</td>
      <td>0.7262</td>
      <td><b>0.6111</b></td>
      <td>0.6623</td>
      <td>0.6355</td>
      <td><b>0.4785</b></td>
      <td>0.5303</td>
      <td>0.7812</td>
      <td><b>0.6920</b></td>
      <td>0.7123</td>
    </tr>
  </tbody>
</table>

**Table 2.** Action-recognition accuracy for OKS Attack.

<table>
  <thead>
    <th>2D HPE model</th>
    <th>Clean acc.</th>
    <th>OKS Attack</th>
    <th>Random noise</th>
  </thead>
  <tbody>
    <tr>
      <td>ResNet-50</td>
      <td>79.03%</td>
      <td>72.85%</td>
      <td>78.06%</td>
    </tr>
    <tr>
      <td>MobileNetV2</td>
      <td>73.52%</td>
      <td>66.30%</td>
      <td>72.38%</td>
    </tr>
    <tr>
      <td>YOLO-Pose M</td>
      <td>76.80%</td>
      <td>69.87%</td>
      <td>72.53%</td>
    </tr>
    <tr>
      <td>YOLO-Pose S</td>
      <td>71.73%</td>
      <td>57.87%</td>
      <td>65.33%</td>
    </tr>
  </tbody>
</table>
</center>


<a id="code-structure"></a>
## 📁 CODE STRUCTURE

This repository is built as a customized extension of the MMPose framework, containing our core modifications for adversarial vulnerability research. To evaluate downstream task performance, we have also integrated the MMAction2 framework directly into our pipeline.

```text
OKS_attack/                   # Root directory (Modified MMPose repository)
├── attacks/                  # Implementations of custom adversarial attacks 
├── demo/                     # Original demos and our modified *_clean.py scripts
├── mmaction2/                # Modified MMAction2 repository
├── mmcv_wheel/               # Custom-build MMCV .whl packages
├── oks_experiments/          # Scripts with executed experiments
├── oks_tools/                # Utility scripts
├── penn-action-dataset/      # Annotations and dataset splits
└── README.md
```

> **Modifications in `demo/`:**
Alongside the original MMPose demo scripts, we introduced custom versions (`topdown_demo_with_mmdet_clean.py` & `bottomup_demo_clean.py`). These modified scripts implement an actor filtering mechanism using the Penn Action `.mat` annotations to effectively remove background individuals, ensuring the pose estimator focuses exclusively on the main actor.

<a id="environment-setup"></a>
## ⚙️ ENVIRONMENT SETUP

### Key Requirements

- **Python** 3.10.18
- **PyTorch** 2.5.1
- **MMCV** 2.1.0
- **CUDA** 12.1

---

### 1. Clone the repository

Before cloning, please install Git LFS (Large File Storage) so the pre-trained model `.pth` checkpoints are downloaded automatically:
```bash
git lfs install
git clone https://github.com/KacperM33/OKS_Attack
cd OKS_Attack/
```

### 2. Create a new Conda environment

This step will install almost all required dependencies, including PyTorch and TorchVision, as defined in the configuration file.
```bash
conda env create -f environment.yml -n oks_attack_env
conda activate oks_attack_env
```
> Note: This process may take a few minutes to complete.

### 3. Install the local MMPose package

Install MMPose framework in editable mode:
```bash
pip install -v -e .
```

### 4. Install MMCV
Depending on your operating system, install MMCV using the pre-built `.whl` files provided in the `mmcv_wheel/` directory.

- For Windows:
```bash
pip install mmcv_wheel/mmcv-2.1.0-cp310-cp310-win_amd64.whl
```

- For Linux:
```bash
pip install mmcv_wheel/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl
```

### 5. Install MMAction2
Navigate to the nested MMAction2 directory and install the local framework in editable mode:
```bash
cd mmaction2/
pip install -e .
cd ..
```

### 6. Install compatible NumPy version
Finally, install a specific NumPy version to ensure compatibility across all integrated frameworks:
```bash
pip install numpy==1.24.4
```

<a id="dataset"></a>
## 🗃️ DATASET

Original dataset: [PENN-ACTION](https://dreamdragon.github.io/PennAction/)

Custom dataset: [PENN-ACTION LOSSLESS VIDEOS MKV DATASET](https://huggingface.co/datasets/urzsi/Penn-Action-Lossless-MKV-Test-Subset)

> We converted the extracted frames into lossless `.mkv` videos (using the FFV1 codec). This is crucial for evaluating adversarial attacks on video action recognition, as standard video compression artifacts would otherwise destroy the fine adversarial perturbations.

### Installation
Download the custom dataset from Hugging Face and place the extracted `videos` folder directly into `OKS_attack/penn-action-dataset/` folder.

<a id="pre-trained-models-checkpoints"></a>
## 🧠 PRE-TRAINED MODELS (CHECKPOINTS)

All required pre-trained model weights are tracked using Git LFS. If you followed the environment setup instructions (`git lfs install` before cloning), these `.pth` files were downloaded automatically.

The following pre-trained models from OpenMMLab are utilized in our pipeline:

| Task | Model Architecture | Checkpoint Path |
| :--- | :--- | :--- |
| **Object Detection** | Faster RCNN (ResNet50)| `checkpoints/faster_rcnn_r50.pth` | 
| **Two-stage Pose Estimation** | ResNet50 | `checkpoints/res50.pth` |
| **Two-stage Pose Estimation** | MobileNetV2 | `checkpoints/mobilenetv2.pth` |
| **Single-stage Pose Estimation** | YOLO-Pose S | `checkpoints/yoloxpose_s.pth` |
| **Single-stage Pose Estimation** | YOLO-Pose M | `checkpoints/yoloxpose_m.pth` |
| **Action Recognition** | SlowOnly | `mmaction2/checkpoints/slowonly_r50_k400-keypoint.pth` |

<a id="tutorial"></a>
## 📓 STEP-BY-STEP TUTORIAL

For a hands-on introduction, we provide an interactive Jupyter Notebook: [`demo_experiment.ipynb`](oks_experiments/demo_experiment.ipynb). This notebook demonstrates how to run the complete adversarial attack pipeline on a single video step by step.

<!-- ## ATTACKS

JAKIE ATAKI -->

<a id="reproducing-experiments"></a>
## 🧪 REPRODUCING EXPERIMENTS

### 1. Generating a dataset subset

If you want to create a custom subset of the dataset for new experiments, use the `select_videos.py` script. 

> **Reproducing paper results:** To reproduce the exact results reported in our paper, **you can skip this step**. We used a 35% subset of the test data divided into 28 batches. The pre-calculated `.txt` splits are already provided in the repository under `penn-action-dataset/subset_selected_files_35p_28b/`.

Example command for generating a custom subset (navigate to the `oks_tools/` directory first):

```bash
python select_videos.py --labels ../penn-action-dataset/labels --output ../penn-action-dataset/subset_selected_files_35p_28b --batches 28 --percent 35
```
Arguments:

- `--labels` - Directory containing .mat annotation files from the Penn Action dataset.

- `--output` - Directory where the generated .txt files (containing sampled video names) will be saved.

- `--batches` - Number of batches to divide the sampled subset into (default: 1).

- `--percent` - Percentage (%) of the entire test set to sample (default: 10).

### 2. Running the experiment

To evaluate the models and run adversarial attacks on the selected videos, use the main experiment script.

Example command (navigate to the `oks_experiments/` directory first):
```bash
python det_oracle_bbox_experiment.py --input ../penn-action-dataset/subset_selected_files_35p_28b/selected_files_batch_0.txt --output experiments_test --model res50
```
Arguments:

- `--input` - A .txt file containing the names of the sampled files for testing (obtained from `select_videos.py` or the provided paper splits).

- `--output` - Directory where the experiment results and logs will be saved.

- `--model` - The pose estimation model architecture to be used for the experiment.

### 3. Available Experiment Scripts

 | Script |  Estimation type | Attack | Ablation study |
 | :--- | :--- | :--- | :--- |
 | `det_oracle_bbox_experiment.py` | two-stage | `attacks/OKS_attack_det.py` | - |
 | `oneshot_experiment.py` | single-stage | `attacks/OKS_attack_oneshot.py` | - |
 | `random_oneshot_experiment.py` | single-stage | `attacks/Random_attack_oneshot.py` | - |
 | `no_det_oracle_bbox_experiment.py` | two-stage | `attacks/OKS_attack_no_det.py` | ☑️ | 
 | `no_det_clean_bbox_experiment.py` | two-stage | `attacks/OKS_attack_no_det.py` | ☑️ | 

<a id="citation"></a>
## ✏️ CITATION (IN PROGRESS)

<!-- If you find this code or our custom OKS attack useful in your research, please consider citing our work:

```bibtex
@inproceedings{mroczek2026oksattack,
  title={A Black-Box Adversarial Attack on Human Pose Estimation and Keypoint-Based Action Recognition Models},
  author={Kacper Mroczek and Michal Kepski},
  booktitle={Advanced Concepts for Intelligent Vision Systems (ACIVS)},
  year={2026},
  note={To appear}
}
``` -->

<a id="acknowledgements"></a>
## 🌐 ACKNOWLEDGEMENTS

- Our custom **OKS attack** is an adaptation inspired by the [**IoU Attack**](https://arxiv.org/pdf/2103.14938)[1].
- This repository is built heavily upon the open-source frameworks provided by [**OpenMMLab**](https://openmmlab.com/codebase). We thank the authors for their open-source contributions.
- We gratefully acknowledge Polish high-performance computing infrastructure **PLGrid (HPC Center: ACK Cyfronet AGH)** for providing
computer facilities and support within computational grant no. PLG/2026/019605.

<br>

> [1] Jia, S., Song, Y., Ma, C., & Yang, X. (2021, June). Iou attack: Towards temporally coherent black-box adversarial attack for visual object tracking. In *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 6705-6714). IEEE.
