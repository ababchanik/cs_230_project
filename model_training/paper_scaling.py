# paper_scaling.py - CORRECTED VERSION
import numpy as np
import torch
import pickle
import os

class PaperScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.is_fitted = False
    
    def fit(self, X):
        """Fit scaler to ALL training data."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        # Store global min/max
        self.data_min_ = X.min(axis=0, keepdims=True)
        self.data_max_ = X.max(axis=0, keepdims=True)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Handle constant features
        self.data_range_[self.data_range_ == 0] = 1.0
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Scale using GLOBAL min/max to [0, 1]."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted")
        
        if isinstance(X, torch.Tensor):
            device = X.device
            dtype = X.dtype
            X_np = X.detach().cpu().numpy()
            return_tensor = True
        else:
            X_np = X
            return_tensor = False
        
        # Scale to [0, 1] using GLOBAL parameters
        X_scaled = (X_np - self.data_min_) / self.data_range_
        
        # Apply feature range if needed
        if self.feature_range != (0, 1):
            min_val, max_val = self.feature_range
            X_scaled = X_scaled * (max_val - min_val) + min_val
        
        if return_tensor:
            return torch.tensor(X_scaled, dtype=dtype, device=device)
        return X_scaled
    
    def inverse_transform(self, X_scaled):
        """Inverse scale using GLOBAL min/max."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted")
        
        if isinstance(X_scaled, torch.Tensor):
            device = X_scaled.device
            dtype = X_scaled.dtype
            X_np = X_scaled.detach().cpu().numpy()
            return_tensor = True
        else:
            X_np = X_scaled
            return_tensor = False
        
        # If scaled from different range, rescale to [0,1] first
        if self.feature_range != (0, 1):
            min_val, max_val = self.feature_range
            X_np = (X_np - min_val) / (max_val - min_val)
        
        # Convert back using GLOBAL parameters
        X_original = X_np * self.data_range_ + self.data_min_
        
        if return_tensor:
            return torch.tensor(X_original, dtype=dtype, device=device)
        return X_original
    
    def save(self, path):
        """Save scaler state."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'data_min_': self.data_min_,
                'data_max_': self.data_max_,
                'data_range_': self.data_range_,
                'feature_range': self.feature_range,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load scaler state."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        scaler = cls(feature_range=data['feature_range'])
        scaler.data_min_ = data['data_min_']
        scaler.data_max_ = data['data_max_']
        scaler.data_range_ = data['data_range_']
        scaler.is_fitted = data['is_fitted']
        
        return scaler