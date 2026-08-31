import json
import cv2
import os
import subprocess
import sys

from argparse import ArgumentParser
from pathlib import Path

# random seed
tools_dir = "../oks_tools"
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

from utils_seed import set_deterministic_seed

set_deterministic_seed(42)

"""
Script for overlaying two skeletons on a single video for comparison:
    - original skeleton - green
    - adversarial skeleton - red
"""

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    return data

def draw_skeletons(color, video_path, json_path, output_path, point=3, line=1):
    data = load_json(json_path)
    frames_data = data["instance_info"]
    skeleton_links = data["meta_info"]["skeleton_links"]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(frames_data):
            break

        persons = frames_data[frame_idx]["instances"]
        for p_idx, person in enumerate(persons):
            kpts = person["keypoints"]

            # draw points
            for x, y in kpts:
                cv2.circle(frame, (int(x), int(y)), point, color, -1)

            # draw links: skeleton_links from meta_info
            for i, j in skeleton_links:
                if i < len(kpts) and j < len(kpts):
                    x1, y1 = kpts[i]
                    x2, y2 = kpts[j]
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, line)#1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--video',
        type=str,
        default='',
        help='original video')
    parser.add_argument(
        '--org_keypoints',
        type=str,
        default='',
        help='json file with keypoints of original video')
    parser.add_argument(
        '--adv_keypoints',
        type=str,
        default='',
        help='json file with keypoints of adversarial video')
    parser.add_argument(
        '--output_path',
        type=str,
        default='',
        help='path to folder for output testing video')
    parser.add_argument(
        '--point',
        type=int,
        default=3,
        help='size of keypoint')
    parser.add_argument(
        '--line',
        type=int,
        default=1,
        help='thickness of line')
    


    args = parser.parse_args()

    if args.video == '':
        raise ValueError('video must not be empty, use --video path_to_original_video.mkv')
    else:
        video = Path(args.video)

    if args.org_keypoints== '':
        raise ValueError('org_keypoints must not be empty, use --org_keypoints path_to_original_keypoints.json')
    else:
        org_keypoints = args.org_keypoints

    if args.adv_keypoints == '':
        raise ValueError('adv_keypoints must not be empty, use --adv_keypoints path_to_adversarial_keypoints.json')
    else:
        adv_keypoints = args.adv_keypoints

    if args.output_path == '':
        raise ValueError('output_path must not be empty, use --output_path path_to_output_folder.json')
    else:
        output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_video = str(Path(f"{output_path}") / (video.stem + "_test" + video.suffix))

    point = args.point
    line = args.line

    # use 1 - drawing original skeleton (GREEN)
    draw_skeletons(
        color = (0, 255, 0),
        video_path = video,  # original video
        json_path = org_keypoints,  # original predicted keypoints
        output_path = f"{output_path}/tmp_green.mkv",  # output tmp video path
        point = point, # point thickness
        line = line # line thickness
    )

    # use 2 - drawing adversarial skeleton (RED)
    draw_skeletons(
        color = (0, 0, 255),
        video_path=f"{output_path}/tmp_green.mkv",  # tmp video with green skeleton
        json_path = adv_keypoints,  # adversarial predicted keypoints
        output_path = output_video,  # output video path
        point = point, # point thickness
        line = line # line thickness
    )

    # tmp video remove
    tmp_path = f"{output_path}/tmp_green.mkv"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"Created visualization of original (green) vs adversarial (red) keypoints in {output_video}.\n")

if __name__ == '__main__':
    main()