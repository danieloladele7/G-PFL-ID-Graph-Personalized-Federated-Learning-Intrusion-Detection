# src/utils/model_utils.py (NEW)
import torch
from pathlib import Path
from typing import Dict, Any

def load_model_from_checkpoint(checkpoint_path: Path, model_class, model_args: Dict):
    """Load model from checkpoint with flexible format handling"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = model_class(**model_args)
        
        if 'model_state_dict' in checkpoint:
            # Standard state dict format
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'parameters' in checkpoint:
            # Flower parameters format (list of numpy arrays)
            from .training_utils import set_model_parameters
            set_model_parameters(model, checkpoint['parameters'])
        else:
            # Assume it's a direct state dict
            model.load_state_dict(checkpoint)
        
        return model, checkpoint
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        return None, None

def save_model_simple(model, save_path: Path, metadata: Dict = None):
    """Simple model saving that always works"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    
    torch.save(save_data, save_path)
    print(f"✅ Model saved: {save_path}")
