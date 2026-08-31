import cv2
import os
import argparse
import numpy as np

def extract_frames_and_noise(original_video_path, adversarial_video_path, frame_number, output_dir='.'):
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_adv = cv2.VideoCapture(adversarial_video_path)

    if not cap_orig.isOpened():
        print(f"Error. Failed to open file {original_video_path}")
        return
    if not cap_adv.isOpened():
        print(f"Error. Failed to open file {adversarial_video_path}")
        return
    
    # positioning at a specific frame
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    cap_adv.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret_orig, frame_orig = cap_orig.read()
    ret_adv, frame_adv = cap_adv.read()

    if not ret_orig:
        print(f"Error while trying to read frame {frame_number} in video {original_video_path}")
        return
    if not ret_adv:
        print(f"Error while trying to read frame {frame_number} in video {adversarial_video_path}")
        return
    
    if frame_adv.shape != frame_orig.shape:
        frame_adv = cv2.resize(frame_adv, (frame_orig.shape[1], frame_orig.shape[0]))

    diff = frame_adv.astype(np.float32) - frame_orig.astype(np.float32)

    multiplier = 5
    diff = diff * multiplier

    noise_image = diff + 128

    noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)

    os.makedirs(output_dir, exist_ok=True)

    orig_out = os.path.join(output_dir, f"frame_{frame_number}_original_new.png")
    adv_out = os.path.join(output_dir, f"frame_{frame_number}_adversarial_new.png")
    noise_out = os.path.join(output_dir, f"frame_{frame_number}_noise_new.png")

    cv2.imwrite(orig_out, frame_orig)
    cv2.imwrite(adv_out, frame_adv)
    cv2.imwrite(noise_out, noise_image)

    print(f"Images saved to directory: {output_dir}")
    print(f" - {orig_out}")
    print(f" - {adv_out}")
    print(f" - {noise_out}")

    cap_orig.release()
    cap_adv.release()


def extract_keypoints(video_keypoints_path, frame_number, output_dir='.'):

    cap = cv2.VideoCapture(video_keypoints_path)

    if not cap.isOpened():
        print(f"Error. Failed to open file {video_keypoints_path}")
        return

    # positioning at a specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        print(f"Error while trying to read frame {frame_number} in video {video_keypoints_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    out = os.path.join(output_dir, f"frame_{frame_number}_keypoints.png")

    cv2.imwrite(out, frame)

    print(f" - {out}")

    cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video',
        type=str,
        default='',
        help="Video number (id)"
    )
    parser.add_argument(
        '--exp',
        type=str,
        default='',
        help="Folder of experiment"
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=0,
        help="Frame number"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help="Output folder"
    )

    args = parser.parse_args()

    if args.video == '':
        raise ValueError('Choose video number (id) in XXXX format, use --video video_number')
    else:
        video = args.video

    if args.exp == '':
        raise ValueError('Experiment folder must not be empty, use --exp experiment_path')
    else:
        exp_folder = args.exp

    FRAME = args.frame
    output_main = args.output

    # ORIG_VIDEO = f'../penn-action-dataset/videos/{video}-vis.mkv'
    # ADV_VIDEO = f'{exp_folder}/{video}-vis_exp/adv/{video}-vis_adv_clean.mkv'
    
    # ORIG_VIDEO = f'{exp_folder}/{video}-vis_exp/og/{video}-vis.mkv'
    ORIG_VIDEO = f'../experiments/{video}-visHYBRID/og/{video}-vis.mkv'
    ADV_VIDEO = f'{exp_folder}/{video}-vis_exp/adv/{video}-vis_adv.mkv'

    OUTPUT = f'{output_main}/extracted_frames/{video}_new'

    KEYPOINTS_VIS = f'{exp_folder}/{video}-vis_exp/results/{video}-vis_test.mkv'
    # KEYPOINTS_VIS = f'{exp_folder}/{video}-vis_exp/adv/{video}-vis_adv_clean.mkv'

    extract_frames_and_noise(ORIG_VIDEO, ADV_VIDEO, FRAME, OUTPUT)
    extract_keypoints(KEYPOINTS_VIS, FRAME, OUTPUT)

'''
example of use:

python extract_frames.py --video 0382 --exp ../experiments_res50_det_oracle --frame 36 --output ../experiments_res50_det_oracle

python extract_frames.py --video ID --exp ../experiments_test --frame XX --output ../experiments_test
'''

if __name__ == "__main__":
    main()