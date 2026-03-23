"""
Verify that all data required for the 23-exercise rehab pipeline is present.
Run after training or anytime to audit:
  - data/unified/<exercise>/  (or data/custom): .npy sequences per exercise
  - data/reference/<exercise>_reference.npy: reference sequence per exercise
  - models/rehab_model.keras, models/meta.json with all 23 exercises

Usage (from project root):
    python scripts/verify_data.py
"""

import os
import sys
import json
import glob

# Project root = parent of scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

EXERCISE_LIST = [
    "squat", "deep_squat", "hurdle_step", "inline_lunge", "side_lunge",
    "sit_to_stand", "standing_leg_raise", "shoulder_abduction",
    "shoulder_extension", "shoulder_rotation", "shoulder_scaption",
    "hip_abduction", "trunk_rotation", "leg_raise", "reach_and_retrieve",
    "wall_pushup", "heel_raise", "bird_dog", "glute_bridge", "clamshell",
    "chin_tuck", "marching_in_place", "step_up",
]

DATA_UNIFIED = "data/unified"
DATA_CUSTOM  = "data/custom"
REF_DIR      = "data/reference"
MODEL_PATH   = "models/rehab_model.keras"
META_PATH    = "models/meta.json"
SEQUENCE_LENGTH = 60
INPUT_DIM = 99


def main():
    data_dir = DATA_UNIFIED if os.path.isdir(DATA_UNIFIED) else DATA_CUSTOM
    print("=" * 60)
    print("  Data & model verification (23 exercises)")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")
    print(f"Reference directory: {REF_DIR}\n")

    all_ok = True

    # 1) Sequence data per exercise
    print("[1] Sequence data (per exercise)")
    print("-" * 50)
    for ex in EXERCISE_LIST:
        ex_dir = os.path.join(data_dir, ex)
        npy_files = glob.glob(os.path.join(ex_dir, "*.npy")) if os.path.isdir(ex_dir) else []
        count = len(npy_files)
        status = "OK" if count > 0 else "MISSING"
        if count == 0:
            all_ok = False
        print(f"  {ex:25} {count:5} sequences  [{status}]")
    print()

    # 2) Reference sequences
    print("[2] Reference sequences (data/reference)")
    print("-" * 50)
    for ex in EXERCISE_LIST:
        ref_path = os.path.join(REF_DIR, f"{ex}_reference.npy")
        if not os.path.isfile(ref_path):
            print(f"  {ex:25} MISSING  (no {ex}_reference.npy)")
            all_ok = False
            continue
        try:
            import numpy as np
            ref = np.load(ref_path)
            if ref.ndim == 3:
                ref = ref.reshape(ref.shape[0], -1)
            if ref.shape[0] < 10 or ref.shape[1] != INPUT_DIM:
                print(f"  {ex:25} INVALID  (shape {ref.shape})")
                all_ok = False
            else:
                print(f"  {ex:25} OK  (shape {ref.shape})")
        except Exception as e:
            print(f"  {ex:25} ERROR  ({e})")
            all_ok = False
    print()

    # 3) Trained model
    print("[3] Trained model & meta")
    print("-" * 50)
    if os.path.isfile(MODEL_PATH):
        print(f"  Model:  {MODEL_PATH}  [OK]")
    else:
        print(f"  Model:  {MODEL_PATH}  [MISSING]")
        all_ok = False

    if os.path.isfile(META_PATH):
        try:
            with open(META_PATH) as f:
                meta = json.load(f)
            meta_ex = meta.get("exercises", [])
            missing = [e for e in EXERCISE_LIST if e not in meta_ex]
            if missing:
                print(f"  Meta:   {META_PATH}  [INCOMPLETE]  missing: {missing}")
                all_ok = False
            else:
                print(f"  Meta:   {META_PATH}  [OK]  ({len(meta_ex)} exercises)")
        except Exception as e:
            print(f"  Meta:   {META_PATH}  [ERROR]  {e}")
            all_ok = False
    else:
        print(f"  Meta:   {META_PATH}  [MISSING]")
        all_ok = False
    print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("  Result: ALL CHECKS PASSED")
    else:
        print("  Result: SOME CHECKS FAILED (see above)")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
