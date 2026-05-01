import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.data_utils import get_train_val_split

def verify_leakage(config_path="configs/advanced.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    train_dir = config["data"]["train_dir"]
    seed = config["training"]["seed"]
    
    print(f"Checking for leakage with seed: {seed}...")
    train_dicts, val_dicts = get_train_val_split(train_dir, val_ratio=0.2, seed=seed)
    
    # Extract IDs (Assuming folder names are unique IDs)
    train_ids = set([os.path.basename(os.path.dirname(d["t1c"])) for d in train_dicts])
    val_ids = set([os.path.basename(os.path.dirname(d["t1c"])) for d in val_dicts])
    
    # 1. Intersection Check
    overlap = train_ids.intersection(val_ids)
    
    print("\n" + "="*40)
    print("      DATA LEAKAGE REPORT")
    print("="*40)
    print(f"Total Unique Patients: {len(train_ids) + len(val_ids)}")
    print(f"Training Patients:      {len(train_ids)}")
    print(f"Validation Patients:    {len(val_ids)}")
    print(f"Overlap (Leakage):      {len(overlap)}")
    
    if len(overlap) == 0:
        print("\n✅ RESULT: ZERO LEAKAGE DETECTED.")
        print("Your validation scores are genuine!")
    else:
        print("\n❌ ALERT: LEAKAGE DETECTED!")
        print(f"Overlapping IDs: {list(overlap)[:5]}...")
    print("="*40)

if __name__ == "__main__":
    verify_leakage()
