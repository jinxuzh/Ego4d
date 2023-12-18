import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import cv2
import time
import hydra
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from ego4d.internal.human_pose.config import Config
from ego4d.research.readers import PyAvReader

from iopath.common.file_io import PathManager
from iopath.common.s3 import S3PathHandler
from projectaria_tools.core import data_provider
from tqdm.auto import tqdm

pathmgr = PathManager()
pathmgr.register_handler(S3PathHandler(profile="default"))


@dataclass
class Context:
    data_dir: str
    repo_root_dir: str
    cache_dir: str
    cache_rel_dir: str
    metadata_json: str
    dataset_dir: str
    dataset_json_path: str
    dataset_rel_dir: str
    frame_dir: str
    ego_cam_names: List[str]
    exo_cam_names: List[str]
    bbox_dir: str
    vis_bbox_dir: str
    pose2d_dir: str
    vis_pose2d_dir: str
    pose3d_dir: str
    vis_pose3d_dir: str
    detector_config: str
    detector_checkpoint: str
    pose_config: str
    pose_checkpoint: str
    dummy_pose_config: str
    dummy_pose_checkpoint: str
    hand_pose_config: str
    hand_pose_ckpt: str
    human_height: float = 1.5
    human_radius: float = 0.3
    min_bbox_score: float = 0.7
    pose3d_start_frame: int = 0
    pose3d_end_frame: int = -1
    refine_pose3d_dir: Optional[str] = None
    vis_refine_pose3d_dir: Optional[str] = None
    take: Optional[Dict[str, Any]] = None
    all_cams: Optional[List[str]] = None
    frame_rel_dir: str = None
    storage_level: int = 30


def mode_preprocess_aria(config: Config):
    data_dir = config.data_dir
    take_json_path = os.path.join(config.data_dir, "takes.json")
    takes = json.load(open(take_json_path))
    take = [t for t in takes if t["root_dir"] == config.inputs.take_name]
    if len(take) != 1:
        print(f"Take: {config.inputs.take_name} does not exist")
        sys.exit(1)
    take = take[0]
    # Get frame_dir
    cache_rel_dir = take["root_dir"]
    cache_dir = os.path.join(
        config.cache_root_dir,
        cache_rel_dir,
    )
    dataset_dir = cache_dir
    frame_dir = os.path.join(dataset_dir, "frames")
    assert config.mode_preprocess.download_video_files, "must download files"

    # Manually selected frames (from annotation file)
    selected_frames_json_path = os.path.join(
        config.inputs.selected_frames_dir, f"{config.inputs.take_name}.json"
    )
    assert os.path.exists(selected_frames_json_path), f"Missing selected frame JSON file for {config.inputs.take_name}"
    selected_frames = json.load(open(selected_frames_json_path))

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
    rel_path = take["frame_aligned_videos"][ego_cam_names]['rgb']["relative_path"]
    path = os.path.join(data_dir, "takes", take["root_dir"], rel_path)
    # Image output directory
    cam_frame_dir = os.path.join(frame_dir, f"{ego_cam_names}_rgb")
    os.makedirs(cam_frame_dir, exist_ok=True)

    reader = PyAvReader(
        path=path,
        resize=None,
        mean=None,
        frame_window_size=1,
        stride=1,
        gpu_idx=-1,
        )
    sample_frames = selected_frames

    # Extract frames
    for idx in tqdm(sample_frames):
        out_path = os.path.join(cam_frame_dir, f"{idx:06d}.jpg")
        if not os.path.exists(out_path):
            frame = reader[idx][0].cpu().numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            assert cv2.imwrite(out_path, frame), out_path


def add_arguments(parser):
    parser.add_argument("--config-name", default="georgiatech_covid_02_2")
    parser.add_argument(
        "--config_path", default="configs", help="Path to the config folder"
    )
    parser.add_argument(
        "--take_name",
        default="georgiatech_covid_02_2",
        type=str,
        help="take names to run, concatenated by '+', "
        + "e.g., uniandes_dance_007_3+iiith_cooking_23+nus_covidtest_01",
    )
    parser.add_argument(
        "--steps",
        default="hand_pose3d_egoexo",
        type=str,
        help="steps to run concatenated by '+', e.g., preprocess+bbox+pose2d+pose3d",
    )


def config_single_job(args, job_id):
    args.job_id = job_id
    args.name = args.name_list[job_id]
    args.work_dir = args.work_dir_list[job_id]
    args.output_dir = args.work_dir

    args.take_name = args.take_name_list[job_id]


def create_job_list(args):
    args.take_name_list = args.take_name.split("+")

    args.job_list = []
    args.name_list = []

    for take_name in args.take_name_list:
        name = take_name
        args.name_list.append(name)
        args.job_list.append(name)

    args.job_num = len(args.job_list)


def parse_args():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    print(args)

    return args


def get_hydra_config(args):
    # https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    hydra.initialize(config_path=args.config_path)
    cfg = hydra.compose(
        config_name=args.config_name,
        # args.opts contains config overrides, e.g., ["inputs.from_frame_number=7000",]
        overrides=args.opts + [f"inputs.take_name={args.take_name}"],
    )
    return cfg


def main(args):
    # Note: this function is called from launch_main.py
    config = get_hydra_config(args)

    steps = args.steps.split("+")
    print(f"steps: {steps}")

    for step in steps:
        print(f"[Info] Running step: {step}")
        start_time = time.time()
        if step == 'preprocess_aria':
            mode_preprocess_aria(config)
        else:
            raise Exception(f"Unknown step: {step}")
        print(f"[Info] Time for {step}: {time.time() - start_time} s")
