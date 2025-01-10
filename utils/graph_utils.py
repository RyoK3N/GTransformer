import numpy as np
import networkx as nx
from dataset.skeleton import Skeleton

def adj_mx_from_skeleton(skeleton: Skeleton) -> np.ndarray:
    num_joints = len(skeleton.joint_names)
    adj = np.zeros((num_joints, num_joints), dtype=np.float32)
    
    # Add connections from skeleton
    for child, parent in skeleton.get_connection_indices():
        adj[child, parent] = 1.0
        adj[parent, child] = 1.0  # Symmetric connections
    
    # Add self-connections
    np.fill_diagonal(adj, 1.0)
    
    # Normalize adjacency matrix
    deg = np.sum(adj, axis=1)
    deg_inv = np.power(deg, -0.5)
    deg_inv[np.isinf(deg_inv)] = 0
    norm_adj = np.multiply(np.multiply(adj, deg_inv[:, np.newaxis]), deg_inv[np.newaxis, :])
    
    return norm_adj

def get_spatial_graph(skeleton: Skeleton, strategy: str = 'distance') -> np.ndarray:
    num_joints = len(skeleton.joint_names)
    adj = np.zeros((num_joints, num_joints), dtype=np.float32)
    
    if strategy == 'connectivity':
        # Use skeleton connectivity
        for child, parent in skeleton.get_connection_indices():
            adj[child, parent] = 1.0
            adj[parent, child] = 1.0
    else:  # distance-based
        # Create fully connected graph with distance-based weights
        for i in range(num_joints):
            for j in range(i + 1, num_joints):
                # Check if joints are connected in skeleton
                connected = any((i, j) == (c, p) or (j, i) == (c, p) 
                              for c, p in skeleton.get_connection_indices())
                if connected:
                    adj[i, j] = adj[j, i] = 1.0
                else:
                    # Add weaker connections for non-directly connected joints
                    adj[i, j] = adj[j, i] = 0.1
    
    # Add self-connections
    np.fill_diagonal(adj, 1.0)
    
    # Normalize adjacency matrix
    deg = np.sum(adj, axis=1)
    deg_inv = np.power(deg, -0.5)
    deg_inv[np.isinf(deg_inv)] = 0
    norm_adj = np.multiply(np.multiply(adj, deg_inv[:, np.newaxis]), deg_inv[np.newaxis, :])
    
    return norm_adj  
