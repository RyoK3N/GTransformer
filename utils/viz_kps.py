import matplotlib.pyplot as plt
import numpy as np
from dataset.skeleton import Skeleton

def visualize_predictions(outputs, targets, return_fig=False, title=None):
    """Visualize predicted camera matrices vs targets"""
    fig = plt.figure(figsize=(10, 5))
    plt.plot(outputs[0], label='Predicted', marker='o')
    plt.plot(targets[0], label='Target', marker='x')
    plt.title(title or 'Camera Matrix Prediction')
    plt.legend()
    plt.grid(True)
    
    if return_fig:
        return fig
    plt.show()
    plt.close()

def visualize_keypoint_skeleton(keypoints, skeleton, return_fig=False):
    """Visualize skeleton with keypoints"""
    fig = plt.figure(figsize=(10, 10))
    keypoints_2d = keypoints.reshape(-1, 2)
    
    # Plot connections
    for child, parent in skeleton.get_connection_indices():
        plt.plot([keypoints_2d[child, 0], keypoints_2d[parent, 0]],
                [keypoints_2d[child, 1], keypoints_2d[parent, 1]],
                'b-', alpha=0.6)
    
    # Plot joints
    plt.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], c='red')
    plt.title('Skeleton Visualization')
    plt.grid(True)
    
    if return_fig:
        return fig
    plt.show()
    plt.close()

def visualize_graph_convolutions(skeleton, activations, writer, global_step):
    """Visualize graph convolutions and attention maps"""
    try:
        # Plot base skeleton structure
        if 'input' in activations:
            input_keypoints = activations['input'][0].cpu().numpy()
            fig = skeleton.plot_graph_with_keypoints(input_keypoints)
            plt.title("Graph Structure with Input Keypoints")
            writer.add_figure('Graph/Structure', fig, global_step)
            plt.close(fig)
        
        # Plot KPA attention maps if available
        if 'after_kpa' in activations:
            kpa_act = activations['after_kpa'][0].cpu().numpy()
            fig = plt.figure(figsize=(15, 5))
            
            # Plot attention heatmap
            plt.subplot(131)
            attention_map = kpa_act @ kpa_act.T
            plt.imshow(attention_map, cmap='viridis')
            plt.colorbar()
            plt.title('KPA Attention Map')
            
            # Plot feature correlation
            plt.subplot(132)
            feature_corr = np.corrcoef(kpa_act.T)
            plt.imshow(feature_corr, cmap='coolwarm')
            plt.colorbar()
            plt.title('Feature Correlations')
            
            # Plot feature activations
            plt.subplot(133)
            plt.imshow(kpa_act, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('Feature Activations')
            
            plt.tight_layout()
            writer.add_figure('Graph/KPAAnalysis', fig, global_step)
            plt.close(fig)
    except Exception as e:
        print(f"Warning: Error in visualize_graph_convolutions: {str(e)}")