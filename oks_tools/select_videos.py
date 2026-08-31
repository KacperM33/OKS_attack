import numpy as np
import scipy
import math
import random
import sys

from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

# random seed
tools_dir = "../oks_tools"
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from utils_seed import set_deterministic_seed

set_deterministic_seed(42)

# load .mat files
def load_classes_from_mat(path):
    folder_path = Path(path)
    sorted_mat = sorted(folder_path.glob('*.mat'))

    test_idx = []
    class_files = defaultdict(list)

    for i, x in enumerate(sorted_mat):
        data = scipy.io.loadmat(x)

        if data['train'] == -1:
            test_idx.append(i+1)

            action = data['action']
            if isinstance(action, np.ndarray) and action.size > 0:
                action_name = str(action.item() if action.size == 1 else action[0])
            else:
                action_name = str(action)

            class_files[action_name].append(x.stem)


    print("\n----- Classes in dataset -----")
    print(f"Number of test videos: {len(test_idx)}")
    for action, files in class_files.items():
        count = len(files)
        print(f"Class {action}: {count} videos")

    return class_files


# draw videos
def select_random(class_files, percent_of_samples=0.10):
    random.seed(42)

    sampled_class_files = {}
    all_files = []

    print(f"----- Drawing {(percent_of_samples*100):.0f}% samples -----\n")

    for action, files in class_files.items():

        sample_size = math.ceil(len(files) * percent_of_samples) # % from every class

        sampled_files = random.sample(files, sample_size)

        sampled_class_files[action] = sampled_files

    for action, files in sampled_class_files.items():
        file_count = len(files)

        print(f"CLass {action} -> {file_count} files")
        print(f"Files {files}\n")
        all_files.extend(files)

    print(f"\nTotal number of files: {len(all_files)}")

    return all_files


# save files
def save_files(all_files, output_path, number_of_batches=4):
    log_path_dir = Path(output_path)
    log_path_dir.mkdir(parents=True, exist_ok=True)

    batch_size = math.ceil(len(all_files) / number_of_batches)

    print(f"Size of single batch: {batch_size}\n")

    # all
    with open(f'{log_path_dir}/selected_files_all.txt', "w") as f:
        for file_name in all_files:
            f.write(f"{file_name}\n")
        print(f"All files (size {len(all_files)})")
        print(f" -> File names saved in `{log_path_dir}/selected_files_all.txt`\n")

    # split into batches
    if number_of_batches > 1:
        for i in range(number_of_batches):
            start_idx = i * batch_size
            end_idx = (i+1) * batch_size

            batch = all_files[start_idx:end_idx]
            print(f"Batch {i}: idx {start_idx} - {end_idx} (size {len(batch)})")

            batch_file_name = f'selected_files_batch_{i}.txt'

            with open(f'{log_path_dir}/{batch_file_name}', "w") as f:
                for file_name in batch:
                    f.write(f"{file_name}\n")
                print(f" -> Saved batch {i} to `{log_path_dir}/{batch_file_name}`\n")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--labels', 
        type=str, 
        default='', 
        help='Path to folder with mat files from penn-action dataset')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Path to output folder')
    parser.add_argument(
        '--batches',
        type=int,
        default=1,
        help='Number of batches')
    parser.add_argument(
        '--percent',
        type=int,
        default=10,
        help='Percent of testing dataset to create subset')
    
    args = parser.parse_args()

    if args.labels == '':
        raise ValueError('Labels folder must not be empty, use --labels path_to_label_mat_files')
    else:
        labels_folder_path = args.labels

    if args.output == '':
        output_selected_files_path = ''
    else:
        output_selected_files_path = args.output

    number_of_batches = args.batches # default 1 batch

    percent_of_samples = args.percent * 0.01 # default 10%
    
    # selection and save
    class_files = load_classes_from_mat(labels_folder_path)

    all_files = select_random(class_files=class_files, percent_of_samples=percent_of_samples) 

    save_files(all_files=all_files, output_path=output_selected_files_path, number_of_batches=number_of_batches)


if __name__ == '__main__':
    main()