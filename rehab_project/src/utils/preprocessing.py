"""
Skeleton Preprocessing Utilities
Handles normalization, augmentation, and sequence formatting
"""

import numpy as np
from scipy.spatial.distance import euclidean


# ── MediaPipe BlazePose joint indices ─────────────────────────────────────────
JOINTS = {
    "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

SEQUENCE_LENGTH = 60   # frames per sample
N_JOINTS       = 33
N_COORDS       = 3     # x, y, z
INPUT_DIM      = N_JOINTS * N_COORDS  # 99 — flattened (T, 33, 3) -> (T, 99)


# ── Core helpers ──────────────────────────────────────────────────────────────

def landmarks_to_array(landmarks) -> np.ndarray:
    """Convert MediaPipe landmark list to (33, 3) numpy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def normalize_skeleton(pose: np.ndarray) -> np.ndarray:
    """
    Centre skeleton on hip and scale by torso length.
    pose: (33, 3) or (T, 33, 3)
    """
    single = (pose.ndim == 2)
    if single:
        pose = pose[np.newaxis]          # (1, 33, 3)

    out = pose.copy()
    for i, frame in enumerate(pose):
        hip_center = (frame[JOINTS["left_hip"]] + frame[JOINTS["right_hip"]]) / 2
        mid_shoulder = (frame[JOINTS["left_shoulder"]] + frame[JOINTS["right_shoulder"]]) / 2
        torso = np.linalg.norm(mid_shoulder - hip_center)
        out[i] = (frame - hip_center) / (torso + 1e-6)

    return out[0] if single else out


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle at joint b in degrees (a–b–c)."""
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


# ── Exercise-specific angles ───────────────────────────────────────────────────

def get_exercise_angles(frame: np.ndarray, exercise: str) -> dict:
    """Return named joint angles relevant to the exercise."""
    J = JOINTS
    f = frame

    def ang(a, b, c):
        return calculate_angle(f[a], f[b], f[c])

    angles = {}

    if exercise in ("squat", "deep_squat"):
        angles["left_knee"]  = ang(J["left_hip"],  J["left_knee"],  J["left_ankle"])
        angles["right_knee"] = ang(J["right_hip"], J["right_knee"], J["right_ankle"])
        angles["left_hip"]   = ang(J["left_shoulder"],  J["left_hip"],  J["left_knee"])
        angles["right_hip"]  = ang(J["right_shoulder"], J["right_hip"], J["right_knee"])
        angles["back"]       = ang(J["left_shoulder"],  J["left_hip"],  J["left_ankle"])

    elif exercise in ("lunge", "forward_lunge"):
        angles["left_knee"]  = ang(J["left_hip"],   J["left_knee"],  J["left_ankle"])
        angles["right_knee"] = ang(J["right_hip"],  J["right_knee"], J["right_ankle"])
        angles["torso"]      = ang(J["nose"],        J["left_hip"],   J["left_ankle"])

    elif exercise in ("arm_raise", "shoulder_abduction"):
        angles["left_elbow"]  = ang(J["left_shoulder"],  J["left_elbow"],  J["left_wrist"])
        angles["right_elbow"] = ang(J["right_shoulder"], J["right_elbow"], J["right_wrist"])
        angles["left_shoulder"]  = ang(J["left_hip"],  J["left_shoulder"],  J["left_elbow"])
        angles["right_shoulder"] = ang(J["right_hip"], J["right_shoulder"], J["right_elbow"])

    elif exercise in ("bird_dog", "quadruped"):
        angles["left_hip"]   = ang(J["left_shoulder"],  J["left_hip"],  J["left_knee"])
        angles["right_hip"]  = ang(J["right_shoulder"], J["right_hip"], J["right_knee"])
        angles["back"]       = ang(J["left_shoulder"],  J["left_hip"],  J["right_shoulder"])

    elif exercise in ("bridge", "hip_bridge", "glute_bridge"):
        angles["left_knee"]  = ang(J["left_hip"],  J["left_knee"],  J["left_ankle"])
        angles["right_knee"] = ang(J["right_hip"], J["right_knee"], J["right_ankle"])
        angles["left_hip"]   = ang(J["left_shoulder"], J["left_hip"], J["left_knee"])

    elif exercise == "hurdle_step":
        angles["left_knee"]  = ang(J["left_hip"],  J["left_knee"],  J["left_ankle"])
        angles["right_knee"] = ang(J["right_hip"], J["right_knee"], J["right_ankle"])
        angles["left_hip"]   = ang(J["left_shoulder"], J["left_hip"], J["left_knee"])
        angles["right_hip"]  = ang(J["right_shoulder"], J["right_hip"], J["right_knee"])

    elif exercise in ("inline_lunge", "side_lunge"):
        angles["left_knee"]  = ang(J["left_hip"],   J["left_knee"],  J["left_ankle"])
        angles["right_knee"] = ang(J["right_hip"],  J["right_knee"], J["right_ankle"])
        angles["torso"]      = ang(J["nose"],        J["left_hip"],   J["left_ankle"])

    elif exercise == "sit_to_stand":
        angles["left_knee"]  = ang(J["left_hip"],  J["left_knee"],  J["left_ankle"])
        angles["right_knee"] = ang(J["right_hip"], J["right_knee"], J["right_ankle"])
        angles["left_hip"]   = ang(J["left_shoulder"], J["left_hip"], J["left_knee"])
        angles["right_hip"]  = ang(J["right_shoulder"], J["right_hip"], J["right_knee"])

    elif exercise == "standing_leg_raise":
        angles["left_hip"]   = ang(J["left_shoulder"], J["left_hip"], J["left_knee"])
        angles["right_hip"]  = ang(J["right_shoulder"], J["right_hip"], J["right_knee"])

    elif exercise in ("shoulder_extension", "shoulder_rotation", "shoulder_scaption"):
        angles["left_elbow"]  = ang(J["left_shoulder"],  J["left_elbow"],  J["left_wrist"])
        angles["right_elbow"] = ang(J["right_shoulder"], J["right_elbow"], J["right_wrist"])
        angles["left_shoulder"]  = ang(J["left_hip"],  J["left_shoulder"],  J["left_elbow"])
        angles["right_shoulder"] = ang(J["right_hip"], J["right_shoulder"], J["right_elbow"])

    elif exercise == "hip_abduction":
        angles["left_hip"]   = ang(J["left_shoulder"], J["left_hip"], J["left_knee"])
        angles["right_hip"]  = ang(J["right_shoulder"], J["right_hip"], J["right_knee"])

    elif exercise == "trunk_rotation":
        angles["back"] = ang(J["left_shoulder"], J["left_hip"], J["right_shoulder"])

    elif exercise == "leg_raise":
        angles["left_hip"]   = ang(J["left_shoulder"], J["left_hip"], J["left_knee"])
        angles["right_hip"]  = ang(J["right_shoulder"], J["right_hip"], J["right_knee"])

    elif exercise == "reach_and_retrieve":
        angles["left_shoulder"]  = ang(J["left_hip"],  J["left_shoulder"],  J["left_elbow"])
        angles["right_shoulder"] = ang(J["right_hip"], J["right_shoulder"], J["right_elbow"])
        angles["left_elbow"]  = ang(J["left_shoulder"],  J["left_elbow"],  J["left_wrist"])
        angles["right_elbow"] = ang(J["right_shoulder"], J["right_elbow"], J["right_wrist"])

    elif exercise == "wall_pushup":
        angles["left_elbow"]  = ang(J["left_shoulder"],  J["left_elbow"],  J["left_wrist"])
        angles["right_elbow"] = ang(J["right_shoulder"], J["right_elbow"], J["right_wrist"])

    elif exercise == "heel_raise":
        angles["left_knee"]  = ang(J["left_hip"],  J["left_knee"],  J["left_ankle"])
        angles["right_knee"] = ang(J["right_hip"], J["right_knee"], J["right_ankle"])

    elif exercise == "clamshell":
        angles["left_hip"]   = ang(J["left_shoulder"], J["left_hip"], J["left_knee"])
        angles["right_hip"]  = ang(J["right_shoulder"], J["right_hip"], J["right_knee"])

    elif exercise == "chin_tuck":
        angles["back"] = ang(J["nose"], J["left_shoulder"], J["left_hip"])

    elif exercise in ("marching_in_place", "step_up"):
        angles["left_knee"]  = ang(J["left_hip"],  J["left_knee"],  J["left_ankle"])
        angles["right_knee"] = ang(J["right_hip"], J["right_knee"], J["right_ankle"])
        angles["left_hip"]   = ang(J["left_shoulder"], J["left_hip"], J["left_knee"])
        angles["right_hip"]  = ang(J["right_shoulder"], J["right_hip"], J["right_knee"])

    else:  # generic – use common angles
        angles["left_knee"]     = ang(J["left_hip"],      J["left_knee"],  J["left_ankle"])
        angles["right_knee"]    = ang(J["right_hip"],     J["right_knee"], J["right_ankle"])
        angles["left_shoulder"] = ang(J["left_hip"],      J["left_shoulder"], J["left_elbow"])

    return angles


# ── Error detection ────────────────────────────────────────────────────────────

EXERCISE_THRESHOLDS = {
    "squat": {
        "left_knee":  (85,  130, "Bend left knee more"),
        "right_knee": (85,  130, "Bend right knee more"),
        "back":       (160, 200, "Keep back straight"),
    },
    "deep_squat": {
        "left_knee":  (60,  110, "Bend left knee deeper"),
        "right_knee": (60,  110, "Bend right knee deeper"),
        "back":       (155, 200, "Keep back straight"),
    },
    "lunge": {
        "left_knee":  (80,  100, "Left knee at 90°"),
        "right_knee": (80,  100, "Right knee at 90°"),
        "torso":      (160, 200, "Keep torso upright"),
    },
    "arm_raise": {
        "left_shoulder":  (80,  100, "Raise left arm to shoulder height"),
        "right_shoulder": (80,  100, "Raise right arm to shoulder height"),
    },
    "bird_dog": {
        "back": (155, 200, "Keep back flat"),
    },
    "bridge": {
        "left_knee":  (80,  100, "Bend knees to 90°"),
        "left_hip":   (160, 200, "Raise hips higher"),
    },
    # 17 additional exercises — standard physiotherapy joint angle ranges
    "hurdle_step": {
        "left_knee":  (85,  120, "Step knee over ankle"),
        "right_knee": (85,  120, "Support knee at 90°"),
        "left_hip":   (70,  100, "Hip flexion for step"),
        "right_hip":  (70,  100, "Hip extension behind"),
    },
    "inline_lunge": {
        "left_knee":  (80,  100, "Front knee at 90°"),
        "right_knee": (80,  100, "Back knee toward floor"),
        "torso":      (160, 200, "Keep torso upright"),
    },
    "side_lunge": {
        "left_knee":  (80,  100, "Bend knee over toes"),
        "right_knee": (170, 190, "Straighten other leg"),
        "torso":      (160, 200, "Keep torso upright"),
    },
    "sit_to_stand": {
        "left_knee":  (85,  100, "Knees over ankles"),
        "right_knee": (85,  100, "Knees over ankles"),
        "left_hip":   (80,  100, "Hinge at hips"),
        "right_hip":  (80,  100, "Hinge at hips"),
    },
    "standing_leg_raise": {
        "left_hip":   (75,  95, "Raise leg to 90°"),
        "right_hip":  (75,  95, "Raise leg to 90°"),
    },
    "shoulder_abduction": {
        "left_shoulder":  (80,  100, "Raise arm to shoulder height"),
        "right_shoulder": (80,  100, "Raise arm to shoulder height"),
    },
    "shoulder_extension": {
        "left_shoulder":  (35,  55, "Extend arm behind"),
        "right_shoulder": (35,  55, "Extend arm behind"),
    },
    "shoulder_rotation": {
        "left_shoulder":  (70,  90, "External rotation range"),
        "right_shoulder": (70,  90, "External rotation range"),
    },
    "shoulder_scaption": {
        "left_shoulder":  (75,  95, "Raise arm in scaption plane"),
        "right_shoulder": (75,  95, "Raise arm in scaption plane"),
    },
    "hip_abduction": {
        "left_hip":   (35,  55, "Lift leg out to side"),
        "right_hip":  (35,  55, "Lift leg out to side"),
    },
    "trunk_rotation": {
        "back": (40,  70, "Rotate trunk within range"),
    },
    "leg_raise": {
        "left_hip":   (75,  95, "Raise leg with control"),
        "right_hip":  (75,  95, "Raise leg with control"),
    },
    "reach_and_retrieve": {
        "left_shoulder":  (60,  90, "Reach forward"),
        "right_shoulder": (60,  90, "Reach forward"),
        "left_elbow":     (140, 180, "Extend arm to reach"),
        "right_elbow":    (140, 180, "Extend arm to reach"),
    },
    "wall_pushup": {
        "left_elbow":  (80,  100, "Bend elbows to 90°"),
        "right_elbow": (80,  100, "Bend elbows to 90°"),
    },
    "heel_raise": {
        "left_knee":  (170, 190, "Keep knees straight"),
        "right_knee": (170, 190, "Keep knees straight"),
    },
    "glute_bridge": {
        "left_knee":  (80,  100, "Bend knees to 90°"),
        "left_hip":   (160, 200, "Raise hips to bridge"),
    },
    "clamshell": {
        "left_hip":   (25,  45, "Open knee within range"),
        "right_hip":  (25,  45, "Open knee within range"),
    },
    "chin_tuck": {
        "back": (160, 190, "Keep neck aligned"),
    },
    "marching_in_place": {
        "left_knee":  (75,  95, "Lift knee to 90°"),
        "right_knee": (75,  95, "Lift knee to 90°"),
        "left_hip":   (75,  95, "Hip flexion for march"),
        "right_hip":  (75,  95, "Hip flexion for march"),
    },
    "step_up": {
        "left_knee":  (80,  100, "Drive knee over ankle"),
        "right_knee": (80,  100, "Support leg at 90°"),
        "left_hip":   (75,  95, "Hip drive on step"),
        "right_hip":  (75,  95, "Hip extension push"),
    },
}

IDEAL_ANGLES = {
    "squat":            {"left_knee": 105, "right_knee": 105, "back": 175},
    "deep_squat":       {"left_knee": 85,  "right_knee": 85,  "back": 170},
    "lunge":            {"left_knee": 90,  "right_knee": 90,  "torso": 180},
    "arm_raise":        {"left_shoulder": 90, "right_shoulder": 90},
    "bird_dog":         {"back": 175},
    "bridge":           {"left_knee": 90, "left_hip": 180},
    "hurdle_step":      {"left_knee": 95,  "right_knee": 95,  "left_hip": 85,  "right_hip": 85},
    "inline_lunge":     {"left_knee": 90,  "right_knee": 90,  "torso": 180},
    "side_lunge":       {"left_knee": 90,  "right_knee": 180, "torso": 180},
    "sit_to_stand":     {"left_knee": 92,  "right_knee": 92,  "left_hip": 90,  "right_hip": 90},
    "standing_leg_raise": {"left_hip": 90,  "right_hip": 90},
    "shoulder_abduction":  {"left_shoulder": 90, "right_shoulder": 90},
    "shoulder_extension": {"left_shoulder": 45, "right_shoulder": 45},
    "shoulder_rotation":  {"left_shoulder": 80, "right_shoulder": 80},
    "shoulder_scaption":  {"left_shoulder": 85, "right_shoulder": 85},
    "hip_abduction":    {"left_hip": 45,  "right_hip": 45},
    "trunk_rotation":   {"back": 55},
    "leg_raise":        {"left_hip": 90,  "right_hip": 90},
    "reach_and_retrieve": {"left_shoulder": 75, "right_shoulder": 75, "left_elbow": 160, "right_elbow": 160},
    "wall_pushup":      {"left_elbow": 90, "right_elbow": 90},
    "heel_raise":       {"left_knee": 180, "right_knee": 180},
    "glute_bridge":     {"left_knee": 90, "left_hip": 180},
    "clamshell":        {"left_hip": 35,  "right_hip": 35},
    "chin_tuck":        {"back": 175},
    "marching_in_place": {"left_knee": 90, "right_knee": 90, "left_hip": 90, "right_hip": 90},
    "step_up":          {"left_knee": 90, "right_knee": 90, "left_hip": 85, "right_hip": 85},
}

def detect_errors(angles: dict, exercise: str) -> list:
    """Return list of error feedback strings."""
    thresholds = EXERCISE_THRESHOLDS.get(exercise, {})
    errors = []
    for joint, value in angles.items():
        if joint in thresholds:
            lo, hi, msg = thresholds[joint]
            if not (lo <= value <= hi):
                errors.append(msg)
    return errors


# ── Sequence buffering ─────────────────────────────────────────────────────────

class FrameBuffer:
    """Sliding-window buffer that accumulates normalised skeleton frames."""

    def __init__(self, length: int = SEQUENCE_LENGTH):
        self.length = length
        self.buffer: list = []

    def add(self, frame: np.ndarray):
        """frame: raw (33,3) array from MediaPipe."""
        norm = normalize_skeleton(frame)
        self.buffer.append(norm)
        if len(self.buffer) > self.length:
            self.buffer.pop(0)

    def ready(self) -> bool:
        return len(self.buffer) >= self.length

    def get_sequence(self) -> np.ndarray:
        """Returns (1, SEQUENCE_LENGTH, 99) for model input."""
        arr = np.array(self.buffer[-self.length:], dtype=np.float32)  # (T,33,3)
        return arr.reshape(1, self.length, N_JOINTS * N_COORDS)

    def reset(self):
        self.buffer.clear()


# ── Score smoothing ────────────────────────────────────────────────────────────

class ScoreSmoother:
    def __init__(self, window: int = 10):
        self.window = window
        self._buf: list = []

    def update(self, score: float) -> float:
        self._buf.append(float(score))
        if len(self._buf) > self.window:
            self._buf.pop(0)
        return float(np.mean(self._buf))

    def reset(self):
        self._buf.clear()


# ── Data augmentation ──────────────────────────────────────────────────────────

def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 33, 3)
    Returns augmented copy.
    """
    aug = seq.copy()

    # Gaussian noise
    aug += np.random.normal(0, 0.01, aug.shape)

    # Random scale (0.9 – 1.1)
    scale = np.random.uniform(0.9, 1.1)
    aug *= scale

    # Random horizontal flip (left ↔ right joints)
    if np.random.rand() < 0.5:
        aug[:, :, 0] *= -1   # flip x

    return aug.astype(np.float32)


def generate_augmented_dataset(sequences, labels, n_augments: int = 5):
    """Multiply dataset n_augments times."""
    aug_seqs, aug_labels = list(sequences), list(labels)
    for seq, lbl in zip(sequences, labels):
        for _ in range(n_augments):
            aug_seqs.append(augment_sequence(seq))
            aug_labels.append(lbl)
    return np.array(aug_seqs, dtype=np.float32), np.array(aug_labels, dtype=np.float32)
