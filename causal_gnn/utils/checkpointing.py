"""Model checkpointing utilities."""

import os
import torch
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class ModelCheckpointer:
    """Handles saving and loading model checkpoints."""

    def __init__(self, checkpoint_dir: str, keep_best_k: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_k = keep_best_k
        self.checkpoints = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Any] = None,
        is_best: bool = False,
        metric_value: Optional[float] = None
    ) -> str:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config.to_dict() if hasattr(config, 'to_dict') else config,
        }

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        if metric_value is not None:
            self.checkpoints.append((metric_value, str(checkpoint_path)))
            self._prune_checkpoints()

        metadata_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(convert_to_serializable({
                'epoch': epoch,
                'metrics': metrics,
                'is_best': is_best
            }), f, indent=2)
        
        print(f"Saved checkpoint to {checkpoint_path}")
        return str(checkpoint_path)

    def _prune_checkpoints(self):
        if len(self.checkpoints) <= self.keep_best_k:
            return

        self.checkpoints.sort(key=lambda x: x[0], reverse=True)

        to_remove = self.checkpoints[self.keep_best_k:]
        self.checkpoints = self.checkpoints[:self.keep_best_k]
        
        for _, checkpoint_path in to_remove:
            try:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    metadata_path = checkpoint_path.replace('.pt', '_metadata.json')
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    print(f"Removed checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Error removing checkpoint {checkpoint_path}: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # weights_only=False: our checkpoints contain python dicts with
        # numpy scalars (metrics). Safe here because checkpoints are written
        # by this same process.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint.get('metrics', {})}")
        
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', None)
        }
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        best_path = self.checkpoint_dir / 'best_model.pt'
        if not best_path.exists():
            print("No best checkpoint found")
            return None
        
        return self.load_checkpoint(str(best_path), model, optimizer, device)

    def get_latest_checkpoint(self) -> Optional[str]:
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoints[-1])

    def list_checkpoints(self) -> list:
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return [str(cp) for cp in checkpoints]

