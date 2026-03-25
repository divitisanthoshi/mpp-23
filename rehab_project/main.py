"""
Real-Time Rehabilitation Exercise Monitor
==========================================
Main application — runs webcam inference with:
  • MediaPipe BlazePose skeleton overlay
  • ST-GCN++ quality scoring (model + DTW fallback)
  • Peak-detection repetition counting
  • Joint error detection with colour highlights
  • Exercise selection menu
  • Progress bar & session summary

Usage:
    python main.py
"""

import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import json
import time

# Fix Python path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.preprocessing import (
    landmarks_to_array, normalize_skeleton,
    get_exercise_angles, detect_errors,
    FrameBuffer, ScoreSmoother,
    JOINTS, SEQUENCE_LENGTH, INPUT_DIM,
)
from src.utils.repetition_counter import RepetitionCounter
from src.models.st_gcn import dtw_score, JointAttention

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ── Config ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/rehab_model.keras"
META_PATH  = "models/meta.json"
REF_DIR    = "data/reference"
DEMO_IMG_DIR = "data/demos"  # optional photos: <exercise_id>.jpg or .png

EXERCISES = [
    ("squat",              "Squat",                "1"),
    ("deep_squat",         "Deep Squat",           "2"),
    ("hurdle_step",        "Hurdle Step",          "3"),
    ("inline_lunge",       "Inline Lunge",         "4"),
    ("side_lunge",         "Side Lunge",           "5"),
    ("sit_to_stand",       "Sit to Stand",         "6"),
    ("standing_leg_raise", "Standing Leg Raise",   "7"),
    ("shoulder_abduction", "Shoulder Abduction",   "8"),
    ("shoulder_extension", "Shoulder Extension",   "9"),
    ("shoulder_rotation",  "Shoulder Rotation",    "a"),
    ("shoulder_scaption",  "Shoulder Scaption",    "b"),
    ("hip_abduction",      "Hip Abduction",       "c"),
    ("trunk_rotation",     "Trunk Rotation",       "d"),
    ("leg_raise",          "Leg Raise",            "e"),
    ("reach_and_retrieve", "Reach and Retrieve",   "f"),
    ("wall_pushup",        "Wall Push-up",         "g"),
    ("heel_raise",         "Heel Raise",           "h"),
    ("bird_dog",           "Bird Dog",             "i"),
    ("glute_bridge",       "Glute Bridge",         "j"),
    ("clamshell",          "Clamshell",            "k"),
    ("chin_tuck",          "Chin Tuck",            "l"),
    ("marching_in_place",  "Marching in Place",    "m"),
    ("step_up",            "Step Up",              "n"),
]
TARGET_REPS = 10

# Short "how to perform" for menu (ASCII only for OpenCV display)
EXERCISE_INSTRUCTIONS = {
    "squat":              "Bend knees, hips back; knees over toes.",
    "deep_squat":         "Squat lower than 90 deg; keep back straight.",
    "hurdle_step":        "Step one foot over hurdle; alternate legs.",
    "inline_lunge":       "Front knee 90 deg; back knee toward floor.",
    "side_lunge":         "Step to side, bend knee; other leg straight.",
    "sit_to_stand":       "Stand from chair using legs; sit back down.",
    "standing_leg_raise": "Lift one leg to front or side; hold balance.",
    "shoulder_abduction": "Raise arms out to sides to shoulder height.",
    "shoulder_extension": "Move arms backward behind body; controlled.",
    "shoulder_rotation": "Elbow at 90; rotate arm in/out at shoulder.",
    "shoulder_scaption":  "Raise arms between forward and side (scaption).",
    "hip_abduction":      "Lift leg out to side; keep standing leg stable.",
    "trunk_rotation":     "Rotate upper body left/right; hips stay still.",
    "leg_raise":          "Lying on back, lift one leg up; lower slowly.",
    "reach_and_retrieve": "Reach forward with arm; return; alternate.",
    "wall_pushup":        "Hands on wall; bend elbows, chest to wall.",
    "heel_raise":         "Rise onto toes; lower slowly. Hold for balance.",
    "bird_dog":           "On all fours: extend one arm, opposite leg.",
    "glute_bridge":       "On back, knees bent; lift hips to ceiling.",
    "clamshell":          "On side, knees bent; lift top knee, feet together.",
    "chin_tuck":          "Tuck chin straight back (double-chin); release.",
    "marching_in_place":   "March on spot; lift knees, swing arms.",
    "step_up":            "Step one foot onto platform; drive up; alternate.",
}

# Step-by-step instructions for reference cards (easy to read, no graphs)
EXERCISE_STEPS = {
    "squat":              ["Stand feet shoulder-width.", "Bend knees, push hips back.", "Keep chest up, knees over toes.", "Return to stand."],
    "deep_squat":         ["Same as squat.", "Go lower (thighs below 90 deg).", "Keep back straight.", "Stand back up."],
    "hurdle_step":        ["Step one foot over a low hurdle.", "Place foot in front.", "Step back. Repeat other leg."],
    "inline_lunge":       ["Step one foot forward.", "Lower back knee toward floor.", "Both knees near 90 deg.", "Push back to start."],
    "side_lunge":         ["Step one leg out to the side.", "Bend that knee, other leg straight.", "Return to centre. Repeat other side."],
    "sit_to_stand":       ["Sit on chair. Shift forward.", "Stand up using your legs.", "Sit back down with control. Repeat."],
    "standing_leg_raise": ["Stand on one leg (hold wall if needed).", "Lift other leg to front or side.", "Lower slowly. Repeat."],
    "shoulder_abduction": ["Arms at sides.", "Raise both arms out to sides.", "Lift to shoulder height. Lower slowly."],
    "shoulder_extension": ["Arms at sides or in front.", "Move arms backward.", "Squeeze shoulder blades. Return."],
    "shoulder_rotation":  ["Elbow bent 90 at your side.", "Rotate forearm and shoulder out.", "Return. Repeat other arm."],
    "shoulder_scaption":  ["Raise arms between forward and side.", "Lift to shoulder height.", "Lower slowly."],
    "hip_abduction":      ["Stand or lie on side.", "Lift one leg out to the side.", "Lower with control. Repeat."],
    "trunk_rotation":     ["Stand or sit. Keep hips still.", "Rotate upper body to one side.", "Return. Rotate to other side."],
    "leg_raise":          ["Lie on back. One leg on floor.", "Lift other leg up toward ceiling.", "Lower slowly. Repeat."],
    "reach_and_retrieve": ["Reach one arm forward.", "Return. Repeat other arm."],
    "wall_pushup":        ["Stand arm's length from wall.", "Hands on wall at shoulder height.", "Bend elbows, chest to wall. Push back."],
    "heel_raise":         ["Stand. Hold wall or chair.", "Rise onto toes.", "Lower slowly. Repeat."],
    "bird_dog":           ["On hands and knees.", "Extend one arm forward.", "Extend opposite leg back. Hold. Switch sides."],
    "glute_bridge":       ["Lie on back. Knees bent, feet flat.", "Lift hips toward ceiling.", "Squeeze glutes. Lower. Repeat."],
    "clamshell":          ["Lie on side. Knees bent, feet together.", "Lift top knee up. Keep feet together.", "Lower. Repeat."],
    "chin_tuck":          ["Sit or stand straight.", "Tuck chin straight back (double chin).", "Hold a few seconds. Release."],
    "marching_in_place":  ["March on the spot.", "Lift knees. Swing arms naturally.", "Stay upright."],
    "step_up":            ["Step one foot onto a low step.", "Push through that leg to bring other foot up.", "Step down. Alternate legs."],
}

# Colours (BGR)
CLR_GREEN  = (50,  220,  50)
CLR_YELLOW = (50,  220, 220)
CLR_RED    = (50,   50, 220)
CLR_WHITE  = (240, 240, 240)
CLR_DARK   = (20,   20,  20)
CLR_PANEL  = (30,   30,  30)
CLR_ACCENT = (220, 150,  50)


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model_safe():
    try:
        import tensorflow as tf
        if os.path.isfile(MODEL_PATH):
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects={"JointAttention": JointAttention},
            )
            print(f"[OK] Loaded model from {MODEL_PATH}")
            return model
    except Exception as e:
        print(f"[!] Model load failed: {e}")
    print("[!] Running in DTW-only mode")
    return None


def load_exercises_meta():
    if os.path.isfile(META_PATH):
        with open(META_PATH) as f:
            return json.load(f).get("exercises", [e[0] for e in EXERCISES])
    return [e[0] for e in EXERCISES]


def load_reference(exercise: str) -> np.ndarray | None:
    path = os.path.join(REF_DIR, f"{exercise}_reference.npy")
    if os.path.isfile(path):
        ref = np.load(path)
        if ref.ndim == 3:
            ref = ref.reshape(ref.shape[0], INPUT_DIM)
        return ref
    return None


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_panel(frame, x, y, w, h, alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), CLR_PANEL, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def put_text(frame, text, pos, scale=0.6, color=CLR_WHITE, thick=1, bold=False):
    if bold:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, CLR_DARK, thick + 3)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick)


def put_text_wrapped(frame, text, x, y_start, line_height, max_chars=48, scale=0.5, color=CLR_WHITE, max_lines=3):
    """Draw text wrapped to max_chars per line (space-separated)."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= max_chars:
            current += " " + w
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    for i, line in enumerate(lines[:max_lines]):
        cv2.putText(frame, line, (x, y_start + i * line_height),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, 1)


def wrap_text_to_lines(text, max_chars=40):
    """Return list of lines (space-separated wrap)."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= max_chars:
            current += " " + w
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def build_reference_card(exercise_id: str, label: str, steps=None, width=420, height=360):
    """
    Build a clear, readable reference card: title + numbered steps (large text).
    """
    if steps is None:
        steps = EXERCISE_STEPS.get(exercise_id, ["Perform with control."])
    card = np.zeros((height, width, 3), dtype=np.uint8)
    card[:] = (42, 42, 48)
    cv2.rectangle(card, (0, 0), (width - 1, height - 1), CLR_ACCENT, 2)
    cv2.putText(card, "HOW TO PERFORM", (18, 32), cv2.FONT_HERSHEY_DUPLEX, 0.58, (180, 180, 180), 1)
    cv2.putText(card, label, (18, 72), cv2.FONT_HERSHEY_DUPLEX, 0.82, CLR_ACCENT, 2)
    cv2.line(card, (16, 86), (width - 16, 86), (70, 70, 70), 1)
    y = 118
    for i, step in enumerate(steps[:5], 1):
        line = f"{i}. {step}"
        for sub in wrap_text_to_lines(line, max_chars=48):
            cv2.putText(card, sub, (22, y), cv2.FONT_HERSHEY_DUPLEX, 0.56, CLR_WHITE, 1)
            y += 32
        y += 6
    return card


def load_demo_images():
    """
    Build clear text instruction cards for each exercise (no skeleton/graph images).
    Uses only step-by-step text so anyone can understand. To use your own photos
    instead, add <exercise_id>.jpg into data/demos/ and they will be used.
    """
    demo_imgs = {}
    for eid, label, _ in EXERCISES:
        # Optional: load a real photo if you added one as .jpg (skeleton .pngs are ignored)
        path_photo = None
        for ext in (".jpg", ".jpeg",):
            p = os.path.join(DEMO_IMG_DIR, eid + ext)
            if os.path.isfile(p):
                path_photo = p
                break
        if path_photo:
            img = cv2.imread(path_photo)
            if img is not None:
                demo_imgs[eid] = img
                continue
        # Default: clear text card with numbered steps (no graph)
        steps = EXERCISE_STEPS.get(eid, [EXERCISE_INSTRUCTIONS.get(eid, "Perform with control.")])
        demo_imgs[eid] = build_reference_card(eid, label, steps=steps)
    return demo_imgs


def draw_score_bar(frame, x, y, w, h, score, label="Score"):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(score / 100 * w)
    color = CLR_GREEN if score >= 80 else CLR_YELLOW if score >= 55 else CLR_RED
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), CLR_WHITE, 1)
    put_text(frame, f"{label}: {score:.0f}%", (x + 5, y + h - 4), 0.45, CLR_WHITE)


def draw_progress_bar(frame, x, y, w, h, reps, target):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(min(reps / max(target, 1), 1.0) * w)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), CLR_ACCENT, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), CLR_WHITE, 1)
    put_text(frame, f"Reps: {reps}/{target}", (x + 5, y + h - 4), 0.45, CLR_WHITE)


def draw_skeleton_errors(frame, landmarks, errors, frame_shape):
    """Highlight error joints in red."""
    ERROR_JOINTS_MAP = {
        "knee": ["left_knee", "right_knee"],
        "back": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "arm":  ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
        "hip":  ["left_hip", "right_hip"],
        "torso":["left_shoulder", "left_hip"],
    }
    error_joints = set()
    for err in errors:
        for key, jnames in ERROR_JOINTS_MAP.items():
            if key in err.lower():
                error_joints.update(jnames)

    h, w = frame_shape[:2]
    if landmarks:
        for jname in error_joints:
            idx = JOINTS.get(jname)
            if idx is None:
                continue
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 12, CLR_RED, -1)
            cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)


# ── Session summary screen ─────────────────────────────────────────────────────
def show_summary(frame, exercise_name, total_reps, avg_score):
    overlay = frame.copy()
    cv2.rectangle(overlay, (80, 80), (frame.shape[1] - 80, frame.shape[0] - 80),
                  (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cx = frame.shape[1] // 2
    put_text(frame, "SESSION COMPLETE", (cx - 180, 150), 1.1, CLR_ACCENT, 2, bold=True)
    put_text(frame, f"Exercise : {exercise_name}", (cx - 180, 230), 0.75, CLR_WHITE)
    put_text(frame, f"Total Reps: {total_reps}", (cx - 180, 275), 0.75, CLR_GREEN)
    avg_col = CLR_GREEN if avg_score >= 80 else CLR_YELLOW if avg_score >= 55 else CLR_RED
    put_text(frame, f"Avg Score : {avg_score:.0f}%", (cx - 180, 320), 0.75, avg_col)
    put_text(frame, "Press R to restart  |  Q to quit",
             (cx - 200, frame.shape[0] - 110), 0.65, CLR_WHITE)
    return frame


# ── Main application ───────────────────────────────────────────────────────────
def run():
    model        = load_model_safe()
    ex_meta      = load_exercises_meta()
    mp_pose      = mp.solutions.pose
    mp_drawing   = mp.solutions.drawing_utils
    mp_styles    = mp.solutions.drawing_styles

    # State
    current_ex_idx = 0
    exercise_id, exercise_label, _ = EXERCISES[current_ex_idx]

    buf         = FrameBuffer(SEQUENCE_LENGTH)
    smoother    = ScoreSmoother(window=12)
    rep_counter = RepetitionCounter(exercise_id)
    ref_seq     = load_reference(exercise_id)

    score        = 0.0
    errors       = []
    show_menu    = True
    session_done = False
    score_history = []
    demo_imgs    = load_demo_images()  # photos from data/demos/ or generated reference cards

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    FW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_pose.Pose(
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        model_complexity=1,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # ── Menu overlay ────────────────────────────────────────────────
            if show_menu:
                draw_panel(frame, 0, 0, FW, FH, alpha=0.7)
                put_text(frame, "REHABILITATION EXERCISE MONITOR",
                         (FW // 2 - 330, 45), 0.9, CLR_ACCENT, 2, bold=True)
                put_text(frame, "Select Exercise:", (60, 85), 0.65, CLR_WHITE)

                # Two columns — 12 left, 11 right
                col1 = EXERCISES[:12]
                col2 = EXERCISES[12:]

                for i, (eid, elabel, ekey) in enumerate(col1):
                    marker = ">>" if EXERCISES.index((eid, elabel, ekey)) == current_ex_idx else "  "
                    col = CLR_ACCENT if EXERCISES.index((eid, elabel, ekey)) == current_ex_idx else CLR_WHITE
                    put_text(frame, f"[{ekey}] {marker} {elabel}",
                             (40, 115 + i * 38), 0.60, col)

                for i, (eid, elabel, ekey) in enumerate(col2):
                    marker = ">>" if EXERCISES.index((eid, elabel, ekey)) == current_ex_idx else "  "
                    col = CLR_ACCENT if EXERCISES.index((eid, elabel, ekey)) == current_ex_idx else CLR_WHITE
                    put_text(frame, f"[{ekey}] {marker} {elabel}",
                             (FW // 2 + 20, 115 + i * 38), 0.60, col)

                # Visual reference for selected exercise (right side) - large and readable
                demo = demo_imgs.get(exercise_id)
                if demo is not None:
                    MENU_REF_W, MENU_REF_H = 380, 360
                    demo_r = cv2.resize(demo, (MENU_REF_W, MENU_REF_H))
                    x0 = FW - MENU_REF_W - 25
                    y0 = 70
                    frame[y0:y0 + MENU_REF_H, x0:x0 + MENU_REF_W] = demo_r
                    cv2.rectangle(frame, (x0, y0), (x0 + MENU_REF_W, y0 + MENU_REF_H), CLR_ACCENT, 2)
                    put_text(frame, "How to perform - read before starting", (x0, y0 - 8), 0.55, CLR_ACCENT, bold=True)

                put_text(frame, "Press key to select  |  Same key or ENTER to start  |  Q quit",
                         (FW // 2 - 300, FH - 28), 0.52, CLR_WHITE)
                cv2.imshow("Rehabilitation Monitor", frame)
                key = cv2.waitKey(1) & 0xFF

                for i, (eid, elabel, kchar) in enumerate(EXERCISES):
                    if key == ord(kchar):
                        if current_ex_idx == i:
                            show_menu = False
                            exercise_id, exercise_label, _ = EXERCISES[i]
                            buf.reset(); smoother.reset()
                            rep_counter.set_exercise(exercise_id)
                            ref_seq = load_reference(exercise_id)
                            score = 0.0; errors = []; score_history = []; session_done = False
                        else:
                            current_ex_idx = i
                            exercise_id, exercise_label, _ = EXERCISES[i]

                if key == 13:
                    show_menu = False
                    buf.reset(); smoother.reset()
                    rep_counter.set_exercise(exercise_id)
                    ref_seq = load_reference(exercise_id)
                    score = 0.0; errors = []; score_history = []; session_done = False
                elif key == ord('q'):
                    break
                continue

            # ── Session summary ──────────────────────────────────────────────
            if session_done:
                avg = float(np.mean(score_history)) if score_history else 0.0
                frame = show_summary(frame, exercise_label,
                                     rep_counter._reps, avg)
                cv2.imshow("Rehabilitation Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    session_done = False
                    buf.reset()
                    smoother.reset()
                    rep_counter.reset()
                    score_history = []
                    score = 0.0
                    errors = []
                elif key == ord('q'):
                    break
                elif key == ord('m'):
                    show_menu = True
                continue

            # ── Pose estimation ──────────────────────────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

                raw_arr = landmarks_to_array(result.pose_landmarks.landmark)
                norm    = normalize_skeleton(raw_arr)

                # Buffer & repetition
                buf.add(raw_arr)
                reps = rep_counter.update(norm)

                # Angle-based errors
                angles = get_exercise_angles(norm, exercise_id)
                errors = detect_errors(angles, exercise_id)

                # Draw red circles on bad joints
                draw_skeleton_errors(frame, result.pose_landmarks.landmark,
                                     errors, frame.shape)

                # Score calculation (only while moving — avoids DTW drift when static)
                if buf.ready() and rep_counter.recent_angle_range(20) > 10.0:
                    seq = buf.get_sequence()   # (1, T, 99)

                    if model is not None:
                        try:
                            preds = model.predict(seq, verbose=0)
                            raw_score = float(preds[0][0][0]) * 100
                        except Exception:
                            raw_score = 50.0
                    else:
                        raw_score = 50.0

                    # Blend with DTW if reference exists
                    if ref_seq is not None:
                        dtw = dtw_score(seq[0], ref_seq)
                        raw_score = 0.5 * raw_score + 0.5 * dtw

                    # Apply error penalty
                    penalty = len(errors) * 5
                    raw_score = max(0, raw_score - penalty)
                    score = smoother.update(raw_score)
                    score_history.append(score)

                # Session done?
                if reps >= TARGET_REPS:
                    session_done = True
                    continue

            # ── UI panels ────────────────────────────────────────────────────

            # Reference panel (how to perform) - left side, large and easy to read
            ref_img = demo_imgs.get(exercise_id)
            if ref_img is not None:
                REF_W, REF_H = 420, 360
                ref_display = cv2.resize(ref_img, (REF_W, REF_H))
                rx, ry = 20, 60
                frame[ry:ry + REF_H, rx:rx + REF_W] = ref_display
                cv2.rectangle(frame, (rx, ry), (rx + REF_W, ry + REF_H), CLR_ACCENT, 2)
                put_text(frame, "Reference - follow these steps", (rx, ry - 8), 0.58, CLR_ACCENT, bold=True)

            # Top panel
            draw_panel(frame, 0, 0, FW, 55)
            put_text(frame, f"Exercise: {exercise_label}", (15, 37),
                     0.8, CLR_ACCENT, 2, bold=True)
            phase = rep_counter.get_phase().upper()
            put_text(frame, f"Phase: {phase}", (FW - 200, 37), 0.65, CLR_WHITE)

            # Right panel
            PW, PH = 260, FH
            px = FW - PW
            draw_panel(frame, px, 55, PW, PH - 55)

            score_color = CLR_GREEN if score >= 80 else CLR_YELLOW if score >= 55 else CLR_RED
            put_text(frame, "QUALITY SCORE", (px + 10, 90), 0.6, CLR_WHITE, bold=True)
            put_text(frame, f"{score:.0f}%", (px + 60, 150), 1.6, score_color, 3, bold=True)
            draw_score_bar(frame, px + 10, 160, PW - 20, 20, score)

            put_text(frame, "REPETITIONS", (px + 10, 210), 0.6, CLR_WHITE, bold=True)
            put_text(frame, str(rep_counter._reps), (px + 90, 265), 1.6, CLR_ACCENT, 3, bold=True)
            draw_progress_bar(frame, px + 10, 275, PW - 20, 20,
                              rep_counter._reps, TARGET_REPS)

            put_text(frame, "FEEDBACK", (px + 10, 320), 0.6, CLR_WHITE, bold=True)
            if errors:
                for i, err in enumerate(errors[:4]):
                    put_text(frame, f"- {err}", (px + 10, 348 + i * 28),
                             0.48, CLR_RED)
            else:
                put_text(frame, "Good form!", (px + 10, 348), 0.55, CLR_GREEN)

            # Bottom controls
            draw_panel(frame, 0, FH - 40, FW, 40)
            put_text(frame, "M=Menu   Q=Quit   R=Reset",
                     (15, FH - 12), 0.55, CLR_WHITE)
            put_text(frame, "RED circles = incorrect joints",
                     (FW - 320, FH - 12), 0.55, CLR_RED)

            cv2.imshow("Rehabilitation Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_menu = True
            elif key == ord('r'):
                buf.reset()
                smoother.reset()
                rep_counter.reset()
                score_history = []
                score = 0.0
                errors = []

    # Final summary to terminal
    if score_history:
        avg = float(np.mean(score_history))
        print(f"\n{'='*50}")
        print(f"  Exercise : {exercise_label}")
        print(f"  Reps     : {rep_counter._reps}")
        print(f"  Avg Score: {avg:.1f}%")
        print(f"{'='*50}\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
