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
                 rgb_keypoint_vis_thres=0.3,  
                 radius=3,
                 thickness=4):
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint
        self.return_heatmap = False
        
        # Initialize pose model
        self.pose_estimator = init_pose_estimator(
            self.pose_config, self.pose_checkpoint, device="cuda:0".lower()
        )
        self.pose_estimator.cfg = adapt_mmdet_pipeline(self.pose_estimator.cfg)
        
        # Initialize body pose visualizer
        self.fast_visualizer = FastVisualizer(self.pose_estimator.dataset_meta,
                                              radius=radius,
                                              line_width=thickness,
                                              kpt_thr=rgb_keypoint_vis_thres)


    ####--------------------------------------------------------
    def get_poses2d(self, bboxes, image_name):
        pose_results = inference_topdown(self.pose_estimator, image_name, bboxes)
        data_samples = merge_data_samples(pose_results)

        # Pose2d kpts
        kpts = data_samples.pred_instances.keypoints
        kpts_conf = data_samples.pred_instances.keypoint_scores
        pose2d_result = np.append(kpts, kpts_conf[:,:,None], axis=2) # (N, 3)

        return pose2d_result


    ####--------------------------------------------------------
    def draw_poses2d(self, pose2d_result, image, save_path):
        vis_img = image.copy()
        for curr_pose2d_res in pose2d_result:
            # Create pose2d instance data
            instanceData = InstanceData()
            assert len(curr_pose2d_res.shape) == 3 and curr_pose2d_res.shape[-1] == 3, "body_pose2d_result should be (1,N,3) for one single image"
            instanceData.keypoints = curr_pose2d_res[:,:,:2]
            instanceData.keypoint_scores = curr_pose2d_res[:,:,2]
                
            # Draw pose2d kpts
            self.fast_visualizer.draw_pose(vis_img, instanceData)
            
            # Save visualization
            cv2.imwrite(save_path, vis_img)


    def draw_projected_poses3d(self, pose_results, image, save_path):
        vis_img = image.copy()
        for curr_pose2d_res in pose_results:
            # Create pose2d instance data
            instanceData = InstanceData()
            assert len(curr_pose2d_res.shape) == 2 and curr_pose2d_res.shape[-1] == 3, "pose_results should be (N,3) for one single image"
            instanceData.keypoints = curr_pose2d_res[:,:2][None,:]
            instanceData.keypoint_scores = curr_pose2d_res[:,2][None,:]
            
            # Draw pose2d kpts
            self.fast_visualizer.draw_pose(vis_img, instanceData)
            
            # Save visualization
            cv2.imwrite(save_path, vis_img)
