# scaling_utils.py
import numpy as np
import torch

class StressScaler:
    """
    Handles scaling of stress tensors to [0, 1] range as per paper.
    Computes statistics from training data and applies to all splits.
    """
    def __init__(self):
        self.sig_min = None
        self.sig_max = None
        self.fitted = False
    
    def fit(self, sig_data):
        """
        Compute min/max from training data.
        sig_data: numpy array of shape (..., 6)
        """
        # Compute global min/max across all dimensions except last
        self.sig_min = sig_data.min(axis=(0, 1), keepdims=True)  # (1, 1, 6)
        self.sig_max = sig_data.max(axis=(0, 1), keepdims=True)  # (1, 1, 6)
        
        print(f"[SCALER] Fitted with:")
        print(f"  Min per component: {self.sig_min.flatten()}")
        print(f"  Max per component: {self.sig_max.flatten()}")
        
        self.fitted = True
        return self
    
    def transform(self, sig):
        """
        Scale stress to [0, 1].
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        # Avoid division by zero
        range_vals = self.sig_max - self.sig_min
        range_vals = np.where(range_vals == 0, 1.0, range_vals)
        
        # Scale to [0, 1]
        sig_scaled = (sig - self.sig_min) / range_vals
        
        return sig_scaled
    
    def inverse_transform(self, sig_scaled):
        """
        Convert scaled stress back to original scale.
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        range_vals = self.sig_max - self.sig_min
        range_vals = np.where(range_vals == 0, 1.0, range_vals)
        
        sig_original = sig_scaled * range_vals + self.sig_min
        
        return sig_original
    
    def save(self, path):
        """Save scaler parameters to file."""
        np.savez(path, sig_min=self.sig_min, sig_max=self.sig_max)
        print(f"[SCALER] Saved to {path}")
    
    def load(self, path):
        """Load scaler parameters from file."""
        data = np.load(path)
        self.sig_min = data['sig_min']
        self.sig_max = data['sig_max']
        self.fitted = True
        print(f"[SCALER] Loaded from {path}")
        return self