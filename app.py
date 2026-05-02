# ─────────────────────────────────────────────────────────────────────────────
# PROJECT STRUCTURE NOTES
# ─────────────────────────────────────────────────────────────────────────────
#
# model/keypoint_classifier/
#   keypoint.csv                      → your raw training data (landmarks)
#   keypoint_classifier_label.csv     → class names (A, B, C... one per line)
#   keypoint_classifier.hdf5          → full Keras model (used for retraining only)
#   keypoint_classifier.tflite        → compressed model (used by app.py at runtime)
#   keypoint_classifier.py            → inference wrapper called by app.py
#
# model/point_history_classifier/
#   point_history.csv                 → training data for motion/dynamic gestures
#   point_history_classifier.hdf5     → full Keras model (retraining only)
#   point_history_classifier.tflite   → used by app.py at runtime
#   point_history_classifier.py       → inference wrapper called by app.py
#   (not needed for static ISL alphabets but kept to avoid import errors)
#
# utils/
#   cvfpscalc.py                      → just calculates FPS for the display
#
# keypoint_classification.ipynb       → run this to retrain after collecting data
#
# CONTROLS (while app.py is running):
#   K   → enter logging mode, terminal asks for class name (A, B, HELLO etc.)
#   S   → save current frame landmarks to keypoint.csv
#   N   → back to normal inference mode
#   ESC → quit
#
# FEATURE VECTOR:
#   Always 84 features = 42 (left/only hand) + 42 (right hand or zeros if absent)
# ─────────────────────────────────────────────────────────────────────────────




#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width",  help='cap width',  type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence",  type=int,   default=0.5)
    args = parser.parse_args()
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Feature constants
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_FEATURES = 84          # 21 landmarks × 2 coords × 2 hands
HAND_FEATURES  = 42          # 21 landmarks × 2 coords × 1 hand
PADDING        = [0.0] * HAND_FEATURES   # fills in missing second hand

# ─────────────────────────────────────────────────────────────────────────────
# CSV / label file paths
# ─────────────────────────────────────────────────────────────────────────────
KEYPOINT_CSV   = 'model/keypoint_classifier/keypoint.csv'
LABEL_CSV      = 'model/keypoint_classifier/keypoint_classifier_label.csv'


# ─────────────────────────────────────────────────────────────────────────────
# Class registry — loads existing labels from the label CSV so class IDs are
# stable across sessions.  New class names are appended automatically.
# ─────────────────────────────────────────────────────────────────────────────
def load_class_registry(label_path):
    """Returns {name: id} dict loaded from the label CSV (creates file if absent)."""
    registry = {}
    if os.path.exists(label_path):
        with open(label_path, encoding='utf-8-sig') as f:
            for idx, row in enumerate(csv.reader(f)):
                if row:
                    registry[row[0].strip()] = idx
    return registry


def save_class_registry(label_path, registry):
    """Writes the label CSV in id order."""
    ordered = sorted(registry.items(), key=lambda x: x[1])
    with open(label_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for name, _ in ordered:
            writer.writerow([name])


def get_or_create_class(registry, label_path, name):
    """Returns the integer id for name, creating a new entry if needed."""
    name = name.strip().upper()
    if name not in registry:
        new_id = len(registry)
        registry[name] = new_id
        save_class_registry(label_path, registry)
        print(f"  [Registry] New class '{name}' assigned id {new_id}")
    return registry[name]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = get_args()

    use_brect = True

    # Camera
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier      = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Load label registries
    class_registry = load_class_registry(LABEL_CSV)

    with open('model/point_history_classifier/point_history_classifier_label.csv',
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    # Reload keypoint labels list (used for live inference display)
    def get_keypoint_labels():
        ordered = sorted(class_registry.items(), key=lambda x: x[1])
        return [name for name, _ in ordered]

    # FPS
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Point history
    history_length         = 16
    point_history          = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # ── Logging state ──────────────────────────────────────────────────────
    # mode 0 = normal inference
    # mode 1 = keypoint logging  (press K to enter, S to save each frame)
    mode             = 0
    current_class_id = -1     # set when user picks a class in mode 1
    current_class_name = ""
    saved_count      = 0      # frames saved for current class this session

    print("Controls:")
    print("  K        → enter keypoint-logging mode (you'll be prompted for class name)")
    print("  S        → save current frame's landmarks to CSV (in logging mode)")
    print("  N        → return to normal inference mode")
    print("  ESC      → quit")

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)

        # ── Key handling ───────────────────────────────────────────────────
        if key == 27:   # ESC
            break

        if key == ord('n') or key == ord('N'):
            mode = 0
            current_class_id   = -1
            current_class_name = ""
            saved_count        = 0
            print("Mode → Normal inference")

        if key == ord('k') or key == ord('K'):
            # Ask for class name in the terminal
            cv.destroyAllWindows()   # briefly hide window so terminal is visible
            class_input = input("\nEnter class name (e.g. A, B, HELLO): ").strip().upper()
            if class_input:
                current_class_id   = get_or_create_class(class_registry, LABEL_CSV, class_input)
                current_class_name = class_input
                saved_count        = 0
                mode               = 1
                print(f"  Logging mode ON  →  class '{current_class_name}' (id {current_class_id})")
                print(f"  Press S to save each frame.  Press N to stop.")
            else:
                print("  No class name entered — staying in normal mode.")
            cv.namedWindow('Hand Gesture Recognition')

        # ── Camera capture ─────────────────────────────────────────────────
        ret, image = cap.read()
        if not ret:
            break
        image       = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # ── MediaPipe ──────────────────────────────────────────────────────
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        feature_vector   = None   # will be set if hands are detected
        hand_sign_id     = -1
        brect            = [0, 0, 0, 0]
        handedness_obj   = None
        landmark_list    = None

        if results.multi_hand_landmarks is not None:
            num_hands = len(results.multi_hand_landmarks)

            # ── Single hand ────────────────────────────────────────────────
            if num_hands == 1:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness_obj = results.multi_handedness[0]

                brect         = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed = pre_process_landmark(landmark_list)
                feature_vector = pre_processed + PADDING   # pad to 84

                pre_processed_point_history = pre_process_point_history(
                    debug_image, point_history)

                # Save on S keypress
                if mode == 1 and current_class_id >= 0 and (key == ord('s') or key == ord('S')):
                    log_keypoint(current_class_id, feature_vector)
                    saved_count += 1
                    print(f"  Saved frame {saved_count} for '{current_class_name}'")

                # Inference
                hand_sign_id = keypoint_classifier(feature_vector)

                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history)
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                kp_labels = get_keypoint_labels()
                sign_label = kp_labels[hand_sign_id] if 0 <= hand_sign_id < len(kp_labels) else "?"

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image, brect, handedness_obj, sign_label,
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

            # ── Two hands ──────────────────────────────────────────────────
            else:
                hand_landmarks_list = results.multi_hand_landmarks
                handedness_list     = results.multi_handedness

                paired = list(zip(hand_landmarks_list, handedness_list))
                paired.sort(key=lambda x: x[1].classification[0].label)

                lm_lists = [calc_landmark_list(debug_image, p[0]) for p in paired]

                pre_left  = pre_process_landmark(lm_lists[0])
                pre_right = pre_process_landmark(lm_lists[1])
                feature_vector = pre_left + pre_right   # 84 features

                brect = calc_bounding_rect_combined(debug_image, paired[0][0], paired[1][0])

                pre_processed_point_history = pre_process_point_history(
                    debug_image, point_history)

                # Save on S keypress
                if mode == 1 and current_class_id >= 0 and (key == ord('s') or key == ord('S')):
                    log_keypoint(current_class_id, feature_vector)
                    saved_count += 1
                    print(f"  Saved frame {saved_count} for '{current_class_name}'")

                # Inference
                hand_sign_id = keypoint_classifier(feature_vector)

                point_history.append([0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history)
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                kp_labels  = get_keypoint_labels()
                sign_label = kp_labels[hand_sign_id] if 0 <= hand_sign_id < len(kp_labels) else "?"

                for lm_list in lm_lists:
                    debug_image = draw_landmarks(debug_image, lm_list)

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(
                    debug_image, brect, paired[0][1], sign_label,
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_logging_info(debug_image, fps, mode,
                                        current_class_name, saved_count)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def log_keypoint(class_id, feature_vector):
    os.makedirs(os.path.dirname(KEYPOINT_CSV), exist_ok=True)
    with open(KEYPOINT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([class_id, *feature_vector])


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for landmark in landmarks.landmark:
        lx = min(int(landmark.x * image_width),  image_width  - 1)
        ly = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [[lx, ly]], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_bounding_rect_combined(image, landmarks1, landmarks2):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for landmarks in [landmarks1, landmarks2]:
        for landmark in landmarks.landmark:
            lx = min(int(landmark.x * image_width),  image_width  - 1)
            ly = min(int(landmark.y * image_height), image_height - 1)
            landmark_array = np.append(landmark_array, [[lx, ly]], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks.landmark:
        lx = min(int(landmark.x * image_width),  image_width  - 1)
        ly = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([lx, ly])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0][0], temp[0][1]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat))
    if max_val > 0:
        flat = [n / max_val for n in flat]
    return flat


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for i, point in enumerate(temp):
        if i == 0:
            base_x, base_y = point[0], point[1]
        temp[i][0] = (temp[i][0] - base_x) / image_width
        temp[i][1] = (temp[i][1] - base_y) / image_height
    return list(itertools.chain.from_iterable(temp))


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        connections = [
            (2,3),(3,4),                         # thumb
            (5,6),(6,7),(7,8),                   # index
            (9,10),(10,11),(11,12),              # middle
            (13,14),(14,15),(15,16),             # ring
            (17,18),(18,19),(19,20),             # little
            (0,1),(1,2),(2,5),(5,9),             # palm
            (9,13),(13,17),(17,0),
        ]
        for a, b in connections:
            cv.line(image, tuple(landmark_point[a]), tuple(landmark_point[b]), (0,0,0), 6)
            cv.line(image, tuple(landmark_point[a]), tuple(landmark_point[b]), (255,255,255), 2)

        for idx, lm in enumerate(landmark_point):
            r = 8 if idx in [4,8,12,16,20] else 5
            cv.circle(image, tuple(lm), r, (255,255,255), -1)
            cv.circle(image, tuple(lm), r, (0,0,0), 1)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0,0,0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1]-22), (0,0,0), -1)
    info_text = handedness.classification[0].label
    if hand_sign_text:
        info_text += ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0]+5, brect[1]-4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
    if finger_gesture_text:
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10,60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10,60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1+int(index/2), (152,251,152), 2)
    return image


def draw_logging_info(image, fps, mode, class_name, saved_count):
    # FPS
    cv.putText(image, "FPS:" + str(fps), (10,30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10,30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv.LINE_AA)

    if mode == 1 and class_name:
        # Red banner at bottom so it's obvious you're recording
        h, w = image.shape[:2]
        cv.rectangle(image, (0, h-60), (w, h), (0,0,180), -1)
        cv.putText(image,
                   f"LOGGING: {class_name}   |   Saved: {saved_count}   |   Press S to save, N to stop",
                   (10, h-20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv.LINE_AA)
    else:
        cv.putText(image, "Press K to start logging a new class", (10,90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()
