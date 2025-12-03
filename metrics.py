""" metrics"""

from sklearn.metrics import accuracy_score, log_loss
import numpy as np

def accuracy(y_true, y_pred):
    """Accuracy score."""
    return accuracy_score(y_true, y_pred)

def demographic_parity(y_pred, sensitive):
    """DP gap for race """
    p0 = y_pred[sensitive[:, 0] == 0].mean()
    p1 = y_pred[sensitive[:, 0] == 1].mean()
    return abs(p0 - p1)

def mia_success(model, X_train, y_train, X_test, y_test):
    try:
        train_pred = model.predict_proba(X_train)
        test_pred = model.predict_proba(X_test)
        
        # FIXED: Provide explicit labels to log_loss
        all_labels = np.unique(np.concatenate([y_train, y_test]))
        labels = [0, 1] if len(all_labels) == 2 else all_labels
        
        # Compute losses with explicit labels
        train_loss = np.array([
            log_loss([y_train[i]], [train_pred[i]], labels=labels) 
            for i in range(len(y_train))
        ])
        
        test_loss = np.array([
            log_loss([y_test[i]], [test_pred[i]], labels=labels) 
            for i in range(len(y_test))
        ])
        
        threshold = np.median(train_loss)
        train_hits = (train_loss < threshold).mean()
        test_hits = (test_loss >= threshold).mean()
        
        return (train_hits + test_hits) / 2
        
    except Exception:
        # Fallback: simple confidence-based MIA
        train_conf = model.predict_proba(X_train).max(axis=1)
        test_conf = model.predict_proba(X_test).max(axis=1)
        
        threshold = np.median(train_conf)
        train_hits = (train_conf > threshold).mean()
        test_hits = (test_conf <= threshold).mean()
        
        return (train_hits + test_hits) / 2
