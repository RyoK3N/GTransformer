import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from dotenv import load_dotenv
import argparse
from utils.sanity_checks import sanity_check
from torch.utils.tensorboard import SummaryWriter
from model.model import KTPFormer
import datetime
import math
import matplotlib.pyplot as plt
from torchviz import make_dot
import torchvision
# Local imports
from model.model import KTPFormer
from dataset.mocap_dataset import MocapDataset
from dataset.skeleton import Skeleton
from utils.graph_utils import adj_mx_from_skeleton
from model.loss import weighted_frobenius_loss
from model.weights import initialize_weights
from utils.viz_kps import visualize_predictions, visualize_keypoint_skeleton, visualize_graph_convolutions
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for KTPFormer')
    
    # Training hyperparameters
    parser.add_argument('--random_seed', type=int, default=100,
                        help='Random seed for reproducibility')
    parser.add_argument('--data_fraction', type=float, default=0.06,
                        help='Fraction of data to use for training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of epochs for learning rate warmup')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    # Model saving and data splits
    parser.add_argument('--model_save_path', type=str, 
                        default='./weights/ktpformer_best_model.pth',
                        help='Path to save the model')
    parser.add_argument('--train_size', type=float, default=0.7,
                        help='Fraction of data to use for training')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    
    # Training configuration
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--visualize_every', type=int, default=1,
                        help='Visualize every N epochs')
    
    # Add tensorboard logging argument
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='Directory for tensorboard logs')

    args = parser.parse_args()
    return args

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def visualize_model_graph(model, writer, input_size=(1, 31, 2)):
    """
    Visualize model architecture and activations in TensorBoard.
    """
    try:
        # Create dummy input
        dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
        
        # Add graph to tensorboard
        writer.add_graph(model, dummy_input)
        writer.flush()
        
        # Add model summary as text
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_summary = (
            f"Model Summary:\n"
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,}\n"
            f"Input shape: {input_size}\n"
        )
        writer.add_text('Model/Summary', model_summary)
        
        # Track initial parameter distributions
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data, 0)
            
    except Exception as e:
        print(f"Warning: Failed to visualize model graph: {str(e)}")

def visualize_activations(writer, activations, global_step, prefix=''):
    """Visualize layer activations in tensorboard"""
    for name, activation in activations.items():
        # Histogram of activation values
        writer.add_histogram(f'{prefix}Activations/{name}', activation.flatten(), global_step)
        
        # Statistics
        writer.add_scalar(f'{prefix}Activations/{name}_mean', activation.mean().item(), global_step)
        writer.add_scalar(f'{prefix}Activations/{name}_std', activation.std().item(), global_step)
        
        # If the activation is 3D (batch, joints, features), visualize the feature maps
        if len(activation.shape) == 3:
            feature_maps = activation[0].detach().cpu().numpy()  # Take first batch
            fig, axes = plt.subplots(1, min(4, feature_maps.shape[1]), figsize=(15, 3))
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            for i, ax in enumerate(axes):
                if i < feature_maps.shape[1]:
                    im = ax.imshow(feature_maps[:, i].reshape(-1, 1), cmap='viridis')
                    ax.set_title(f'Feature {i}')
                    plt.colorbar(im, ax=ax)
            plt.tight_layout()
            writer.add_figure(f'{prefix}FeatureMaps/{name}', fig, global_step)
            plt.close(fig)

def visualize_graph_conv(writer, model, keypoints, camera_matrix, skeleton, global_step):
    """Visualize graph convolution operations"""
    model.track_activations = True
    outputs, activations = model(keypoints)
    model.track_activations = False
    
    # Visualize input skeleton
    fig = plt.figure(figsize=(10, 10))
    keypoints_np = keypoints[0].cpu().numpy().reshape(-1, 2)
    
    # Plot connections
    for child, parent in skeleton.get_connection_indices():
        plt.plot([keypoints_np[child, 0], keypoints_np[parent, 0]],
                [keypoints_np[child, 1], keypoints_np[parent, 1]],
                'b-', alpha=0.6)
    
    # Plot joints
    plt.scatter(keypoints_np[:, 0], keypoints_np[:, 1], c='red')
    plt.title('Input Skeleton')
    writer.add_figure('Graph/InputSkeleton', fig, global_step)
    plt.close(fig)
    
    # Visualize KPA attention maps if available
    if 'after_kpa' in activations:
        kpa_activations = activations['after_kpa'][0].cpu().numpy()
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(kpa_activations @ kpa_activations.T, cmap='viridis')
        plt.colorbar()
        plt.title('KPA Attention Map')
        writer.add_figure('Graph/KPAAttention', fig, global_step)
        plt.close(fig)

def compute_frobenius_norm(pred, target):
    """Compute Frobenius norm between prediction and target"""
    return torch.norm(pred - target, p='fro')

def train(args):
    # Set seeds for reproducibility
    set_seeds(args.random_seed)

    # Initialize tensorboard writer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, current_time)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Load environment variables
    load_dotenv()
    uri = os.getenv('URI')
    if not uri:
        raise EnvironmentError("Please set the 'URI' environment variable in your .env file.")

    # Define skeleton structure
    connections = [
        ('Head', 'Neck'), ('Neck', 'Chest'), ('Chest', 'Hips'),
        ('Neck', 'LeftShoulder'), ('LeftShoulder', 'LeftArm'),
        ('LeftArm', 'LeftForearm'), ('LeftForearm', 'LeftHand'),
        ('Chest', 'RightShoulder'), ('RightShoulder', 'RightArm'),
        ('RightArm', 'RightForearm'), ('RightForearm', 'RightHand'),
        ('Hips', 'LeftThigh'), ('LeftThigh', 'LeftLeg'),
        ('LeftLeg', 'LeftFoot'), ('Hips', 'RightThigh'),
        ('RightThigh', 'RightLeg'), ('RightLeg', 'RightFoot'),
        ('RightHand', 'RightFinger'), ('RightFinger', 'RightFingerEnd'),
        ('LeftHand', 'LeftFinger'), ('LeftFinger', 'LeftFingerEnd'),
        ('Head', 'HeadEnd'), ('RightFoot', 'RightHeel'),
        ('RightHeel', 'RightToe'), ('RightToe', 'RightToeEnd'),
        ('LeftFoot', 'LeftHeel'), ('LeftHeel', 'LeftToe'),
        ('LeftToe', 'LeftToeEnd'),
        ('SpineLow', 'Hips'), ('SpineMid', 'SpineLow'), ('Chest', 'SpineMid')
    ]

    joints_left = [
        'LeftShoulder', 'LeftArm', 'LeftForearm', 'LeftHand', 'LeftFinger', 'LeftFingerEnd',
        'LeftThigh', 'LeftLeg', 'LeftFoot', 'LeftHeel', 'LeftToe', 'LeftToeEnd'
    ]

    joints_right = [
        'RightShoulder', 'RightArm', 'RightForearm', 'RightHand', 'RightFinger', 'RightFingerEnd',
        'RightThigh', 'RightLeg', 'RightFoot', 'RightHeel', 'RightToe', 'RightToeEnd'
    ]

    # Initialize dataset
    dataset = MocapDataset(uri=uri, db_name='ai', collection_name='cameraPoses', skeleton=None)
    
    # Setup skeleton
    skeleton = Skeleton(
        connections=connections,
        joints_left=joints_left,
        joints_right=joints_right,
        ordered_joint_names=dataset.joint_names
    )
    dataset.skeleton = skeleton

    # Apply data fraction
    total_samples = len(dataset)
    samples_to_use = int(total_samples * args.data_fraction)
    dataset._ids = dataset._ids[:samples_to_use]
    dataset.total = samples_to_use

    print(f"Using {samples_to_use} samples out of {total_samples}")
    print(f"Number of joints: {dataset.num_joints}")
    print(f"Joint names: {dataset.joint_names}")

    sanity_check(dataset)
    
    split_generator = torch.Generator().manual_seed(args.random_seed)

    train_length = int(args.train_size * len(dataset))
    val_length = int(args.val_size * len(dataset))
    test_length = len(dataset) - train_length - val_length

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_length, val_length, test_length],
        generator=split_generator
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    loader_generator = torch.Generator().manual_seed(args.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  
        pin_memory=True,
        generator=loader_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )

    adj_matrix = adj_mx_from_skeleton(skeleton)
    model = KTPFormer(
        input_dim=dataset.num_joints * 2,
        embed_dim=512,
        adj=adj_matrix,
        depth=4,
        num_heads=8,
        drop_rate=0.2
    ).to(args.device)

    model.apply(initialize_weights)

    # Add model graph visualization
    visualize_model_graph(model, writer, input_size=(1, dataset.num_joints, 2))

    # Track gradients and weights
    for name, param in model.named_parameters():
        writer.add_histogram(f'Parameters/{name}', param.data, 0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    scaler = GradScaler() if args.device == 'cuda' else None

    def validate(epoch, show_visualization=True):
        model.eval()
        val_loss = 0.0
        val_frob_loss = 0.0
        val_recon_loss = 0.0
        
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                keypoints_2d, keypoints_3d, camera_matrix = batch_data[0], batch_data[1], batch_data[2]
                keypoints_2d = keypoints_2d.to(args.device)
                keypoints_3d = keypoints_3d.to(args.device)
                camera_matrix = camera_matrix.to(args.device)
                
                model.track_activations = True
                outputs, activations = model(keypoints_2d)
                
                loss, frob_loss, recon_loss = weighted_frobenius_loss(
                    outputs, 
                    camera_matrix,
                    keypoints_3d.view(keypoints_3d.shape[0], -1, 3),
                    keypoints_2d.view(keypoints_2d.shape[0], -1, 2),
                    alpha=0.5
                )
                
                val_loss += loss.item()
                val_frob_loss += frob_loss.item()
                val_recon_loss += recon_loss.item()
                
                if show_visualization and i % 50 == 0:
                    try:
                        # Validation visualizations
                        fig = skeleton.plot_graph_with_keypoints(keypoints_2d[0].cpu().numpy())
                        writer.add_figure('Validation/InputSkeleton', fig, global_step)
                        plt.close(fig)
                        
                        visualize_graph_convolutions(skeleton, activations, writer, global_step)
                        
                        fig = plt.figure(figsize=(10, 5))
                        pred_np = outputs[0].detach().cpu().numpy()
                        target_np = camera_matrix[0].detach().cpu().numpy()
                        plt.plot(pred_np, label='Predicted', marker='o')
                        plt.plot(target_np, label='Target', marker='x')
                        plt.title(f'Validation Predictions (Batch {i})')
                        plt.legend()
                        plt.grid(True)
                        writer.add_figure('Validation/Predictions', fig, global_step)
                        plt.close(fig)
                        
                        writer.flush()
                    except Exception as e:
                        print(f"Warning: Error in validation visualization: {str(e)}")

        return (val_loss / len(val_loader), 
                val_frob_loss / len(val_loader),
                val_recon_loss / len(val_loader))

    best_val_loss = float('inf')
    no_improvement_count = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_frob_loss = 0.0
        total_recon_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        try:
            for batch_idx, batch_data in enumerate(progress_bar):
                keypoints_2d, keypoints_3d, camera_matrix = batch_data[0], batch_data[1], batch_data[2]
                keypoints_2d = keypoints_2d.to(args.device)
                keypoints_3d = keypoints_3d.to(args.device)
                camera_matrix = camera_matrix.to(args.device)
                
                optimizer.zero_grad()

                # Always track activations
                model.track_activations = True
                outputs, activations = model(keypoints_2d)
                
                # Compute combined loss
                loss, frob_loss, recon_loss = weighted_frobenius_loss(
                    outputs, 
                    camera_matrix,
                    keypoints_3d.view(keypoints_3d.shape[0], -1, 3),
                    keypoints_2d.view(keypoints_2d.shape[0], -1, 2),
                    alpha=0.5  # Adjust this weight as needed
                )

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Log metrics
                writer.add_scalar('Loss/train_step', loss.item(), global_step)
                writer.add_scalar('Loss/frobenius_loss', frob_loss.item(), global_step)
                writer.add_scalar('Loss/reconstruction_loss', recon_loss.item(), global_step)
                writer.add_scalar('Learning_rate/step', scheduler.get_last_lr()[0], global_step)

                total_loss += loss.item()
                total_frob_loss += frob_loss.item()
                total_recon_loss += recon_loss.item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'frob_loss': f'{frob_loss.item():.4f}',
                    'recon_loss': f'{recon_loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })

                # Detailed visualization for every 10 batches 
                if batch_idx % 10 == 0:
                    try:
                        # Input skeleton visualization
                        fig = skeleton.plot_graph_with_keypoints(keypoints_2d[0].cpu().numpy())
                        plt.title(f'Input Skeleton (Epoch {epoch+1}, Batch {batch_idx})')
                        writer.add_figure('Training/InputSkeleton', fig, global_step)
                        plt.close(fig)

                        # Graph convolution visualization
                        visualize_graph_convolutions(skeleton, activations, writer, global_step)

                        # Prediction visualization
                        fig = plt.figure(figsize=(10, 5))
                        pred_np = outputs[0].detach().cpu().numpy()
                        target_np = camera_matrix[0].detach().cpu().numpy()
                        plt.plot(pred_np, label='Predicted', marker='o')
                        plt.plot(target_np, label='Target', marker='x')
                        plt.title(f'Predictions vs Targets (Epoch {epoch+1}, Batch {batch_idx})')
                        plt.legend()
                        plt.grid(True)
                        writer.add_figure('Training/Predictions', fig, global_step)
                        plt.close(fig)

                        # Activation heatmaps
                        for name, activation in activations.items():
                            if len(activation.shape) == 3:  
                                act_np = activation[0].detach().cpu().numpy()
                                fig = plt.figure(figsize=(12, 4))
                                plt.imshow(act_np, cmap='viridis', aspect='auto')
                                plt.colorbar()
                                plt.title(f'{name} Activation Map')
                                writer.add_figure(f'Activations/{name}', fig, global_step)
                                plt.close(fig)

                        # Ensure updates are written
                        writer.flush()
                    except Exception as e:
                        print(f"Warning: Error in visualization: {str(e)}")

                scheduler.step()
                global_step += 1

            # End of epoch logging
            avg_train_loss = total_loss / len(train_loader)
            avg_frob_loss = total_frob_loss / len(train_loader)
            avg_recon_loss = total_recon_loss / len(train_loader)
            
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('Metrics/epoch_frobenius_loss', avg_frob_loss, epoch)
            writer.add_scalar('Metrics/epoch_reconstruction_loss', avg_recon_loss, epoch)
            
            # Validation phase
            val_loss, val_frob_loss, val_recon_loss = validate(epoch, show_visualization=True)
            
            writer.add_scalar('Loss/validation_epoch', val_loss, epoch)
            writer.add_scalar('Metrics/validation_frobenius_norm', val_frob_loss, epoch)
            writer.add_scalar('Metrics/validation_reconstruction_loss', val_recon_loss, epoch)

            # Learning rate tracking
            writer.add_scalar('Learning_rate/epoch', scheduler.get_last_lr()[0], epoch)

            print(f"Epoch {epoch+1}/{args.epochs} Summary:")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Training Frob Loss: {avg_frob_loss:.4f}")
            print(f"  Training Recon Loss: {avg_recon_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Validation Frob Norm: {val_frob_loss:.4f}")
            print(f"  Validation Recon Loss: {val_recon_loss:.4f}")

            # Model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss,
                    'frob_norm': val_frob_loss,
                }, args.model_save_path)
                print(f"  Model saved with validation loss: {best_val_loss:.4f}")
            else:
                no_improvement_count += 1
                if no_improvement_count >= args.early_stop_patience:
                    print(f"No improvement for {args.early_stop_patience} epochs. Early stopping.")
                    break

        except Exception as e:
            print(f"Error in training epoch {epoch+1}: {str(e)}")
            continue

    writer.close()

if __name__ == "__main__":
    args = parse_args()
    train(args)
