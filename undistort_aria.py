import glob
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from projectaria_tools.core import calibration
from ego4d.internal.human_pose.camera import get_aria_camera_models
from ego4d.internal.human_pose.utils import aria_original_to_extracted
import cv2


def find_ego_cam_name(take):
    # Find aria names (some takes don't have aria01 as default)
    ego_cam_names = [
        x["cam_id"] for x in take["capture"]["cameras"] if str(x["is_ego"]).lower() == "true"
    ]
    assert len(ego_cam_names) > 0, "No ego cameras found!"
    if len(ego_cam_names) > 1:
        ego_cam_names = [
            cam for cam in ego_cam_names if cam in take["frame_aligned_videos"].keys()
        ]
        assert len(ego_cam_names) > 0, "No frame-aligned ego cameras found!"
        if len(ego_cam_names) > 1:
            ego_cam_names_filtered = [
                cam for cam in ego_cam_names if "aria" in cam.lower()
            ]
            if len(ego_cam_names_filtered) == 1:
                ego_cam_names = ego_cam_names_filtered
        assert (
            len(ego_cam_names) == 1
        ), f"Found too many ({len(ego_cam_names)}) ego cameras: {ego_cam_names}"

    # Load video reader
    ego_cam_names = ego_cam_names[0]
    return ego_cam_names


def get_interested_take_uid(common_take_uid, takes_df):
    interested_scenarios = ['Health', 'Bike Repair', 'Music', 'Cooking']
    scenario_take_dict = {scenario:[] for scenario in interested_scenarios}
    all_interested_scenario_uid = []
    for curr_local_cam_valid_uid in common_take_uid: # MODIFY
        curr_scenario = takes_df[takes_df['take_uid'] == curr_local_cam_valid_uid]['scenario_name'].item()
        if curr_scenario in interested_scenarios:
            scenario_take_dict[curr_scenario].append(curr_local_cam_valid_uid)
            all_interested_scenario_uid.append(curr_local_cam_valid_uid)
    all_interested_scenario_uid = sorted(all_interested_scenario_uid)
    return all_interested_scenario_uid


def main():
    method = 'automatic' # MODIFY
    DATA_ROOT_DIR = "/mnt/volume2/Data/Ego4D"
    raw_img_root_dir = "/mnt/volume2/Data/Ego4D/aria_raw_images"
    aria_undist_img_root = f"/mnt/volume2/Data/Ego4D/aria_undistorted_images"
    check_common_takes = True       # For all locally available takes, whether only consider those common takes also exist in train/val/test split
    check_exist_cam_pose = True     # For each take, whether check if it has corresponding cam pose file before undistortion
    check_interested_take = True    # Among all common takes, filter based on interested scenarios (e.g. music, cooking)
    os.makedirs(aria_undist_img_root, exist_ok=True)

    # Find all take uid
    hand_anno_dir = os.path.join(DATA_ROOT_DIR, f"annotations/ego_pose/hand/{method}/")
    anno_avail_take_uid = [k[:-5] for k in os.listdir(hand_anno_dir)]
    # Find all takes name
    takes = json.load(open(os.path.join(DATA_ROOT_DIR, "takes.json")))
    take_to_uid = {each_take['root_dir'] : each_take['take_uid'] for each_take in takes if each_take["take_uid"] in anno_avail_take_uid}
    uid_to_take = {uid:take for take, uid in take_to_uid.items()}
    
    # Load train/val/test split
    takes_df = pd.read_csv(os.path.join(DATA_ROOT_DIR, 'annotations/egoexo_split_latest_train_val_test.csv'))
    all_used_uid = list(takes_df['take_uid'])
    # Find common takes between locally available takes and takes in train/val/test split
    common_take_uid = list(set(anno_avail_take_uid) & set(all_used_uid)) if check_common_takes else anno_avail_take_uid
    common_take_uid = sorted(common_take_uid)
    print(f"Got {len(common_take_uid)} common takes with check_cam_pose={check_exist_cam_pose}")

    # Determine undistortion uids
    undist_uid = get_interested_take_uid(common_take_uid, takes_df) if check_interested_take else common_take_uid
    print(f"Among those common takes, find {len(undist_uid)} interested takes. Start undistortion now.")

    # Iterate through all common takes
    undist_idx = 1
    for curr_take_uid in [undist_uid[0]]:
        curr_take_name = uid_to_take[curr_take_uid]
        
        # Input and output directory for current take
        take = [t for t in takes if t["root_dir"] == curr_take_name]
        take = take[0]
        ego_cam_name = find_ego_cam_name(take)
        distorted_img_dir = os.path.join(raw_img_root_dir, f'{curr_take_name}/frames/{ego_cam_name}_rgb')
        curr_aria_undist_img_dir = os.path.join(aria_undist_img_root, curr_take_name)

        # Check if exist original distorted aria images
        if not os.path.exists(distorted_img_dir):
            print(f"[Warning] Original extracted images missing for {curr_take_name}. Skipped for now.")
            continue
        # Check if exists camera pose file
        curr_cam_pose_path = os.path.join(DATA_ROOT_DIR, "annotations/ego_pose/hand/camera_pose", f"{curr_take_uid}.json")
        if check_exist_cam_pose and not os.path.exists(curr_cam_pose_path):
            print(f"[Warning] Camera pose file missing for {curr_take_name}. Skip for now.")
            continue

        # Load aria calibration model
        capture_name = '_'.join(curr_take_name.split('_')[:-1])
        pinhole = calibration.get_linear_camera_calibration(512, 512, 150)
        vrs_path = os.path.join(DATA_ROOT_DIR, 'captures', capture_name, f'videos/{ego_cam_name}.vrs')
        aria_rgb_calib = get_aria_camera_models(vrs_path)['214-1']

        # Load selected frames for current take based on anno_type
        all_frame_number = json.load(open(os.path.join(DATA_ROOT_DIR, f"annotations/ego_pose/hand/selected_frames_info_{method}", f"{curr_take_name}.json")))
        os.makedirs(curr_aria_undist_img_dir, exist_ok=True)

        # Generate undistorted images using frame index from annotation file
        print(f"====== [{undist_idx}] Generating undistorted images for {curr_take_name} with anno_type={method} ======")
        for frame_idx in tqdm(all_frame_number):
            undist_output_img_path = os.path.join(curr_aria_undist_img_dir, f"{frame_idx:06d}.jpg")
            if not os.path.exists(undist_output_img_path):
                distorted_img_path = os.path.join(distorted_img_dir, f"{frame_idx:06d}.jpg")
                image = np.array(Image.open(distorted_img_path).rotate(90))
                # Undistort aria-rgb images
                undistorted_image = calibration.distort_by_calibration(image, pinhole, aria_rgb_calib)
                undistorted_image = cv2.rotate(undistorted_image, cv2.ROTATE_90_CLOCKWISE)
                # Save undistorted images
                cv2.imwrite(undist_output_img_path, undistorted_image[:,:,::-1])
        undist_idx += 1


if __name__ == '__main__':
    main()