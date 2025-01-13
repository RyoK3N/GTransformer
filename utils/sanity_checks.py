import torch
import numpy as np
from tqdm import tqdm

def sanity_check(dataset):
    """
    Perform basic sanity checks on the dataset.
    """
    print("\nPerforming sanity check on 50 samples...")
    for i in tqdm(range(min(50, len(dataset)))):
        try:
            sample = dataset[i]
            
            # Check number of elements in sample
            assert len(sample) == 5, "Sample should contain 5 elements (keypoints_2d, keypoints_3d, camera_matrix, joint_names, indices)"
            
            keypoints_2d, keypoints_3d, camera_matrix, joint_names, indices = sample
            
            # Check keypoints_2d shape
            assert isinstance(keypoints_2d, torch.Tensor), "keypoints_2d should be a torch.Tensor"
            assert keypoints_2d.dim() == 1, "keypoints_2d should be a 1D tensor"
            assert keypoints_2d.size(0) == dataset.num_joints * 2, f"keypoints_2d should have shape ({dataset.num_joints * 2},)"
            
            # Check keypoints_3d shape
            assert isinstance(keypoints_3d, torch.Tensor), "keypoints_3d should be a torch.Tensor"
            assert keypoints_3d.dim() == 1, "keypoints_3d should be a 1D tensor"
            assert keypoints_3d.size(0) == dataset.num_joints * 3, f"keypoints_3d should have shape ({dataset.num_joints * 3},)"
            
            # Check camera matrix shape
            assert isinstance(camera_matrix, torch.Tensor), "camera_matrix should be a torch.Tensor"
            assert camera_matrix.dim() == 1, "camera_matrix should be a 1D tensor"
            assert camera_matrix.size(0) == 16, "camera_matrix should have shape (16,)"
            
            # Check joint names and indices
            assert isinstance(joint_names, list), "joint_names should be a list"
            assert isinstance(indices, list), "indices should be a list"
            assert len(joint_names) == dataset.num_joints, "Number of joint names should match num_joints"
            assert len(indices) == dataset.num_joints, "Number of indices should match num_joints"
            
        except Exception as e:
            print(f"\nSanity check failed at sample {i}: {str(e)}")
            raise
            
    print("Sanity check passed!") 
 
