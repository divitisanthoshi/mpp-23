"""
Generate reference images from the same skeleton data the model was trained on.
Loads data/reference/<exercise>_reference.npy (normalized pose sequences),
renders key frames as stick-figure images, and saves to data/demos/<exercise>.png.
These images match the model's view of the exercise (skeleton space).
"""

import os
import sys
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

REF_DIR = "data/reference"
DEMO_DIR = "data/demos"
INPUT_DIM = 99
N_JOINTS = 33
N_COORDS = 3

# Exercise IDs (same order as in main.py / train.py)
EXERCISE_IDS = [
    "squat", "deep_squat", "hurdle_step", "inline_lunge", "side_lunge",
    "sit_to_stand", "standing_leg_raise", "shoulder_abduction",
    "shoulder_extension", "shoulder_rotation", "shoulder_scaption",
    "hip_abduction", "trunk_rotation", "leg_raise", "reach_and_retrieve",
    "wall_pushup", "heel_raise", "bird_dog", "glute_bridge", "clamshell",
    "chin_tuck", "marching_in_place", "step_up",
]

# BlazePose 33-point skeleton connections (indices) for drawing
# Torso, arms, legs - enough to show pose clearly
SKELETON_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (9, 10),
]


def seq_to_frames(seq: np.ndarray):
    """(T, 99) -> list of (33, 3) per frame."""
    if seq.ndim == 2:
        T = seq.shape[0]
        seq = seq.reshape(T, N_JOINTS, N_COORDS)
    return [seq[t] for t in range(seq.shape[0])]


def draw_skeleton_frame(frame_pose: np.ndarray, img_size=(320, 240), margin=0.15):
    """
    frame_pose: (33, 3) in normalized coords (hip-centered, torso-scaled).
    Returns BGR image with stick figure. img_size = (width, height).
    """
    w, h = img_size[0], img_size[1]
    # Use x, y only; normalize to image with margin
    xy = frame_pose[:, :2].copy()
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    span = max(xmax - xmin, ymax - ymin, 0.5) * (1 + margin)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    xy[:, 0] = (xy[:, 0] - cx) / span * (w * 0.45) + w / 2
    xy[:, 1] = (xy[:, 1] - cy) / span * (h * 0.45) + h / 2
    # Y is up in skeleton, down in image
    xy[:, 1] = h - xy[:, 1]

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (35, 35, 40)
    joint_color = (220, 180, 100)
    bone_color = (180, 140, 80)
    joint_rad = 3
    thick = 2

    for (i, j) in SKELETON_EDGES:
        if 0 <= i < len(xy) and 0 <= j < len(xy):
            pt1 = (int(round(xy[i, 0])), int(round(xy[i, 1])))
            pt2 = (int(round(xy[j, 0])), int(round(xy[j, 1])))
            cv2.line(img, pt1, pt2, bone_color, thick)
    for i in range(len(xy)):
        pt = (int(round(xy[i, 0])), int(round(xy[i, 1])))
        cv2.circle(img, pt, joint_rad, joint_color, -1)
    return img


def make_reference_image(seq: np.ndarray, img_size=(320, 240), n_frames=3):
    """
    Combine n_frames (start, mid, end) into one horizontal strip or single representative frame.
    seq: (T, 99) or (T, 33, 3)
    """
    frames = seq_to_frames(seq)
    T = len(frames)
    if T == 0:
        return np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    # Pick key frames
    indices = [0, T // 2, T - 1] if T >= 3 else [T // 2]
    indices = indices[:n_frames]
    imgs = [draw_skeleton_frame(frames[i], img_size) for i in indices]
    if len(imgs) == 1:
        return imgs[0]
    # Horizontal layout: 3 panels (w, h = width, height)
    w, h = img_size[0], img_size[1]
    strip = np.zeros((h, w * len(imgs), 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        strip[:, i * w:(i + 1) * w] = im
    # Resize back to single image size so it fits in the app
    out = cv2.resize(strip, (w, h))
    return out


def main():
    os.makedirs(DEMO_DIR, exist_ok=True)
    print("Generating reference images from model training data (data/reference/*.npy)")
    print("-" * 60)
    for ex_id in EXERCISE_IDS:
        path = os.path.join(REF_DIR, f"{ex_id}_reference.npy")
        if not os.path.isfile(path):
            print(f"  [SKIP] {ex_id}: no reference file")
            continue
        seq = np.load(path)
        if seq.ndim == 3:
            seq = seq.reshape(seq.shape[0], INPUT_DIM)
        img = make_reference_image(seq, img_size=(320, 240), n_frames=3)
        out_path = os.path.join(DEMO_DIR, f"{ex_id}.png")
        cv2.imwrite(out_path, img)
        print(f"  [OK]   {ex_id} -> {out_path}")
    print("-" * 60)
    print(f"Done. Images saved to {DEMO_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
