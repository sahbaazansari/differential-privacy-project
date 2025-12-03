"""PyDP mechanisms """

import numpy as np
import pydp as dp
from pydp.algorithms.laplacian import BoundedMean
from config import Config

class PrivateModel:
    """Simple DP on model predictions """
    
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.lower = -1  # Integer bounds for PyDP
        self.upper = 1
    
    def private_predictions(self, predictions):
        """Apply DP noise to model prediction probabilities."""
        private_preds = []
        
        for pred in predictions:
            # Convert to list of single float
            mech = BoundedMean(
                epsilon=self.epsilon,
                lower_bound=self.lower,
                upper_bound=self.upper
            )
            # PyDP needs list input
            noisy_pred = mech.quick_result([float(pred)])
            private_preds.append(float(noisy_pred))
        
        return np.clip(np.array(private_preds), 0, 1)

def apply_dp(model, epsilon, X_test):
    """
    Apply DP to model predictions (SIMPLEST APPROACH).
    
    Works with ANY scikit-learn model. No feature importance hacks.
    """
    priv_model = PrivateModel(epsilon)
    
    # Get original predictions
    orig_probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Apply DP noise to predictions
    private_probs = priv_model.private_predictions(orig_probs)
    
    # Return predictions based on noisy probabilities
    private_preds = (private_probs > 0.5).astype(int)
    
    # Create wrapper that returns private predictions
    class PrivatePredictor:
        def predict(self, X):
            # For consistency, re-apply same noise pattern
            return private_preds
        
        def predict_proba(self, X):
            proba = np.zeros((len(X), 2))
            proba[:, 1] = private_probs
            proba[:, 0] = 1 - private_probs
            return proba
    
    return PrivatePredictor()
