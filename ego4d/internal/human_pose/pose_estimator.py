import os

import cv2
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.visualization import FastVisualizer
from mmengine.structures import InstanceData

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
        self.fast_visualizer = FastVisualizer(self.body_pose_estimator.dataset_meta,
                                              radius=radius,
                                              line_width=thickness,
                                              kpt_thr=rgb_keypoint_vis_thres)


    ####--------------------------------------------------------
    def get_poses2d(self, bboxes, image_name):
        body_pose_results = inference_topdown(self.body_pose_estimator, image_name, bboxes)
        body_data_samples = merge_data_samples(body_pose_results)

        # Body pose2d kpts
        body_kpts = body_data_samples.pred_instances.keypoints
        body_kpts_conf = body_data_samples.pred_instances.keypoint_scores
        body_pose2d_result = np.append(body_kpts, body_kpts_conf[:,:,None], axis=2) # (N, 3)

        return body_pose2d_result


    ####--------------------------------------------------------
    def draw_poses2d(self, body_pose2d_result, image, save_path):
        # Create pose2d instance data
        instanceData = InstanceData()
        assert len(body_pose2d_result.shape) == 3 and body_pose2d_result.shape[-1] == 3, "body_pose2d_result should be (1,N,3) for one single image"
        instanceData.keypoints = body_pose2d_result[:,:,:2]
        instanceData.keypoint_scores = body_pose2d_result[:,:,2]
        
        # Draw pose2d kpts
        vis_img = image.copy()
        self.fast_visualizer.draw_pose(vis_img, instanceData)
        
        # Save visualization
        cv2.imwrite(save_path, vis_img)

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
    def draw_projected_poses3d(self, pose_results, image, save_path):
        # Create pose2d instance data
        instanceData = InstanceData()
        assert len(pose_results.shape) == 2 and pose_results.shape[-1] == 3, "pose_results should be (N,3) for one single image"
        instanceData.keypoints = pose_results[:,:2][None,:,:]
        instanceData.keypoint_scores = pose_results[:,2][None,:,:]
        
        # Draw pose2d kpts
        vis_img = image.copy()
        self.fast_visualizer.draw_pose(vis_img, instanceData)
        
        # Save visualization
        cv2.imwrite(save_path, vis_img)
