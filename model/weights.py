import torch
import torch.nn as nn
import math

def initialize_weights(m: nn.Module) -> None:
    """
    Initialize model weights using Xavier/Glorot initialization.
    
    Args:
        m (nn.Module): Model layer to initialize
    """
    if isinstance(m, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
    elif isinstance(m, nn.LayerNorm):
        # Initialize LayerNorm with ones 
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
            
    elif isinstance(m, nn.Parameter):
        # Initialize other parameters with Xavier/Glorot
        nn.init.xavier_uniform_(m) 
