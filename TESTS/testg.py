import json
import numpy as np
from pathlib import Path

BASE = Path("data/processed")
INDEX = BASE / "bm25_index"

files = [
    INDEX / "data.csc.index.npy",
]

for path in files:
    print(f"\n{'=' * 60}")
    print(f"FILE: {path}")
    print("=" * 60)

    try:
        if path.suffix == ".npy":
            data = np.load(path, allow_pickle=False)
            print("Shape:", data.shape)
            print("Dtype:", data.dtype)
            print("Content:")
            print(data)


    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}")