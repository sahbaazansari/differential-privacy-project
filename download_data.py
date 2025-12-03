"""Download UCI Adult dataset"""

import requests
import pandas as pd
from pathlib import Path

def download_uci_adult():
    """Download and save UCI Adult dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    output_path = Path("adult.data")
    
    if output_path.exists():
        print(f"✓ Dataset already exists: {output_path}")
        return str(output_path)
    
    print("Downloading UCI Adult dataset...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    print(f"✓ Downloaded to: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    download_uci_adult()
