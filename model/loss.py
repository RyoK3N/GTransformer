import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weighted_frobenius_loss(pred: torch.Tensor, target: torch.Tensor, points_3d: torch.Tensor, points_2d: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """
    Frobenius norm and reconstruction error
    
    Args:
        pred: Predicted camera matrix (B, 16)
        target: Target camera matrix (B, 16)
        points_3d: 3D keypoints (B, N, 3)
        points_2d: 2D keypoints (B, N, 2)
        alpha: Weight for combining losses (default: 0.7)
    """
    # Frobenius norm loss for ground truth camera matrices vs predicted camera matrices
    pred_matrix = pred.view(-1, 4, 4)
    target_matrix = target.view(-1, 4, 4)
    frob_norm = torch.norm(pred_matrix - target_matrix, p='fro', dim=(1, 2))
    frob_loss = torch.mean(frob_norm)
    
    # Reconstruction loss for ground truth 2D points vs predicted 2D points
    batch_size = pred.shape[0]
    recon_loss = 0.0
    
    for i in range(batch_size):
        # Convert tensors to numpy for existing DLT functions
        points_3d_np = points_3d[i].detach().cpu().numpy()
        points_2d_np = points_2d[i].detach().cpu().numpy()
        camera_matrix = target_matrix[i].detach().cpu().numpy()
        predicted_camera_matrix = pred_matrix[i].detach().cpu().numpy()
        
        # Project points using predicted camera matrix but for calculating the projection matrix we use the target camera matrix
        proj_matrix = find_projection_matrix_dlt(points_3d_np, points_2d_np, camera_matrix)
        projected_points = project_points_dlt(points_3d_np, predicted_camera_matrix, proj_matrix)
        
        # Convert back to tensors
        projected_points = torch.from_numpy(projected_points).float().to(points_2d.device)
        points_2d_i = points_2d[i].view(-1, 2)
        
        # Compute the reconstruction error using MSE loss
        # Option 1: Try L1 Loss 
        #recon_error = F.l1_loss(projected_points, points_2d_i)
        
        # Option 2: Try Smooth L1 Loss 
        recon_error = F.smooth_l1_loss(projected_points, points_2d_i)
        
        recon_loss += recon_error
    
    recon_loss = recon_loss / batch_size
    
    # Scale the losses 
    #frob_loss = frob_loss * 0.1  # for Scaling down 
    #recon_loss = recon_loss * 10.0  # for Scaling up 
    
    # Combine the losses 
    total_loss = alpha * frob_loss + (1 - alpha) * recon_loss
    
    return total_loss, frob_loss, recon_loss



def normalize_points(points):
    """Normalize points by centering and scaling."""
    mean = np.mean(points, axis=0)
    scale = np.sqrt(2) / np.std(points - mean)
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    return T


def find_projection_matrix_dlt(points_3d, points_2d, view_matrix):
    """Find projection matrix using Direct Linear Transform."""
    # Convert 3D points to camera space
    points_homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    points_camera = points_homogeneous @ view_matrix.T
    
    # Normalize 2D points
    points_2d_homogeneous = np.hstack([points_2d, np.ones((len(points_2d), 1))])
    
    # Build the DLT matrix
    A = []
    for i in range(len(points_3d)):
        X = points_camera[i]
        x = points_2d_homogeneous[i]
        
        A.append([
            X[0], X[1], X[2], X[3], 0, 0, 0, 0, -x[0]*X[0], -x[0]*X[1], -x[0]*X[2], -x[0]*X[3]
        ])
        A.append([
            0, 0, 0, 0, X[0], X[1], X[2], X[3], -x[1]*X[0], -x[1]*X[1], -x[1]*X[2], -x[1]*X[3]
        ])
    
    A = np.array(A)
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    projection_matrix = np.vstack([P, [0, 0, 0, 1]])
    
    return projection_matrix

def project_points_dlt(points_3d, view_matrix, projection_matrix):
    """Project 3D points using the found projection matrix."""
    # Convert to homogeneous coordinates
    points_homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    
    # Transform to camera space
    points_camera = points_homogeneous @ view_matrix.T
    
    # Project points
    points_projected = points_camera @ projection_matrix.T
    
    # Perspective divide
    points_2d = points_projected[:, :2] / points_projected[:, 2:3]
    
    return points_2d


