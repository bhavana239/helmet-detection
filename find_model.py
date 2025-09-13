import os
from pathlib import Path

base_dir = Path.home() / "runs" / "detect"

# List all folders that start with 'train'
folders = [f for f in base_dir.glob("train*") if f.is_dir()]

# Sort by most recently modified
folders.sort(key=os.path.getmtime, reverse=True)

# Check each folder for weights/best.pt
for folder in folders:
    best_pt = folder / "weights" / "best.pt"
    if best_pt.exists():
        print(f"✅ Trained model found in: {folder}")
        break
else:
    print("❌ No trained model found with best.pt.")
