"""Dataset configuration."""

class Config:
    # UCI Adult dataset settings
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    TARGET_COL = "income"  # >50K or <=50K
    SENSITIVE_COLS = ["race", "sex"]  # Protected attributes
    
    # Model settings
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Privacy budgets
    EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # PyDP bounds 
    FEATURE_BOUNDS = (-3.0, 3.0)
