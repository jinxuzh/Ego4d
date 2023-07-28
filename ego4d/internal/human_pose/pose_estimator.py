import os

import cv2
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.registry import VISUALIZERS

##------------------------------------------------------------------------------------
class PoseModel:
    def __init__(self, 
                 pose_config=None, 
                 pose_checkpoint=None, 
                 rgb_keypoint_vis_thres=0.7,  
                 radius=4,
                 thickness=4):
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.return_heatmap = False

        # Initialize pose model
        self.body_pose_estimator = init_pose_estimator(
            self.pose_config, self.pose_checkpoint, device="cuda:0".lower()
        )
        self.body_pose_estimator.cfg = adapt_mmdet_pipeline(self.body_pose_estimator.cfg)
        
        # Initialize body pose visualizer
        self.body_pose_estimator.cfg.visualizer.radius = radius
        self.body_pose_estimator.cfg.visualizer.alpha = 1
        self.body_pose_estimator.cfg.visualizer.line_width = thickness
        self.body_visualizer = VISUALIZERS.build(self.body_pose_estimator.cfg.visualizer)
        self.body_visualizer.set_dataset_meta(
            self.body_pose_estimator.dataset_meta, skeleton_style='mmpose')

        ##------hyperparameters-----
        self.rgb_keypoint_vis_thres = rgb_keypoint_vis_thres  ## Keypoint score threshold



    ####--------------------------------------------------------
    def get_poses2d(self, bboxes, image_name):
        body_pose_results = inference_topdown(self.body_pose_estimator, image_name, bboxes)
        body_data_samples = merge_data_samples(body_pose_results)

        # Body pose2d kpts
        body_kpts = body_data_samples.pred_instances.keypoints
        body_kpts_conf = body_data_samples.pred_instances.keypoint_scores
        body_pose2d_result = np.append(body_kpts, body_kpts_conf[:,:,None], axis=2) # (N, 3)

        return body_pose2d_result, body_data_samples


    ####--------------------------------------------------------
    def draw_poses2d(self, body_data_samples, image, save_path):
        body_pose_vis = self.body_visualizer.add_datasample(
            'result',
            image,
            data_sample=body_data_samples,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=False,
            show_kpt_idx=False,
            skeleton_style='mmpose',
            show=False,
            wait_time=0,
            kpt_thr=self.rgb_keypoint_vis_thres)
        
        # Save visualization
        cv2.imwrite(save_path, body_pose_vis)

    ####--------------------------------------------------------
    def draw_projected_poses3d(self, pose_results, image_name, save_path):
        keypoint_thres = self.rgb_keypoint_vis_thres

        ##-----------restructure to the desired format used by mmpose---------
        pose_results_ = []
        for pose in pose_results:
            pose_ = np.zeros(
                (self.num_keypoints, 3)
            )  ## N x 3 (17 for body; 21 for hand)

            pose_[: len(pose), :3] = pose[:, :]

            pose_result = {"keypoints": pose_}
            pose_results_.append(pose_result)

        pose_results = pose_results_

        vis_pose_result(
            self.pose_model,
            image_name,
            pose_results,
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            kpt_score_thr=keypoint_thres,
            radius=self.radius,
            thickness=self.thickness,
            show=False,
            out_file=save_path,
        )
