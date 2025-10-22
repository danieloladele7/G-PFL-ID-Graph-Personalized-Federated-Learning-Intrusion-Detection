# src/utils/saving_utils.py (NEW)
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

class ResultsManager:
    """Manages saving of results, models, and metrics"""
    
    def __init__(self, experiment_name: str, model_type: str, strategy: str):
        self.experiment_name = experiment_name
        self.model_type = model_type
        self.strategy = strategy
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.base_dir = Path("results") / experiment_name / model_type / strategy
        self.checkpoints_dir = Path("checkpoints") / experiment_name / model_type / strategy
        self.metrics_dir = self.base_dir / "metrics"
        
        for dir_path in [self.base_dir, self.checkpoints_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_training_history(self, history: Dict, round_num: int):
        """Save training history with proper formatting"""
        history_file = self.metrics_dir / f"training_history_round_{round_num}.json"
        
        # Ensure all values are JSON serializable
        serializable_history = self._make_serializable(history)
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2, cls=NumpyEncoder)
        
        print(f"✅ Training history saved: {history_file}")
    
    def save_model_checkpoint(self, model, parameters, round_num: int, is_best: bool = False):
        """Save model checkpoint with metadata"""
        # Convert parameters to state dict
        if hasattr(parameters, 'tensors'):
            from flwr.common import parameters_to_ndarrays
            parameters_ndarrays = parameters_to_ndarrays(parameters)
        else:
            parameters_ndarrays = parameters
        
        # Set parameters and save
        from src.utils import set_model_parameters
        set_model_parameters(model, parameters_ndarrays)
        
        # Save model
        if is_best:
            model_path = self.checkpoints_dir / "global_model_best.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': self.model_type,
                'round': round_num,
                'timestamp': self.timestamp,
                'strategy': self.strategy
            }, model_path)
            print(f"✅ Best model saved: {model_path}")
        else:
            model_path = self.checkpoints_dir / f"global_model_round_{round_num}.pt"
            torch.save(model.state_dict(), model_path)
        
        return model_path
    
    def save_metrics_csv(self, metrics_history: List[Dict], filename: str = "metrics_summary.csv"):
        """Save metrics as CSV for easy analysis"""
        if not metrics_history:
            return
        
        df = pd.DataFrame(metrics_history)
        csv_path = self.metrics_dir / filename
        df.to_csv(csv_path, index=False)
        print(f"✅ Metrics CSV saved: {csv_path}")
        return csv_path
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                             np.int32, np.int64, np.uint8, np.uint16, 
                             np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        else:
            return obj

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                           np.int32, np.int64, np.uint8, np.uint16, 
                           np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
