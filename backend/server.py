from flask import Flask, jsonify, Response
from flask_cors import CORS

import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import copy
import itertools
import threading

from model import KeyPointClassifier

app = Flask(__name__)
CORS(app)

# =========================
# Feature constants
# =========================
HAND_FEATURES = 42
PADDING       = [0.0] * HAND_FEATURES

# =========================
# MediaPipe
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# =========================
# Classifier + Labels
# =========================
keypoint_classifier = KeyPointClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

# =========================
# Webcam
# =========================
cap = cv.VideoCapture(0)

# =========================
# Shared state
# Updated by the background recognition thread
# Read by the frontend via /api/result
# =========================
latest_result = {
    'gesture':    'No hand detected',
    'confidence': 0,
    'text':       ''
}
result_lock = threading.Lock()

# =========================
# Helpers
# =========================
def calc_landmark_list(image, landmarks):
    iw, ih = image.shape[1], image.shape[0]
    pts = []
    for lm in landmarks.landmark:
        pts.append([min(int(lm.x * iw), iw - 1),
                    min(int(lm.y * ih), ih - 1)])
    return pts


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    bx, by = temp[0][0], temp[0][1]
    for i in range(len(temp)):
        temp[i][0] -= bx
        temp[i][1] -= by
    flat = list(itertools.chain.from_iterable(temp))
    mv = max(map(abs, flat))
    if mv > 0:
        flat = [n / mv for n in flat]
    return flat


def get_confidence(feature_vector, predicted_id):
    try:
        interp = keypoint_classifier.interpreter
        ind    = interp.get_input_details()
        outd   = interp.get_output_details()
        interp.set_tensor(ind[0]['index'],
                          np.array([feature_vector], dtype=np.float32))
        interp.invoke()
        probs = interp.get_tensor(outd[0]['index'])[0]
        return int(round(float(probs[predicted_id]) * 100))
    except Exception:
        return 95


# =========================
# Background thread
# Reads webcam, runs MediaPipe + classifier,
# stores latest result in latest_result dict.
# This is EXACTLY what app.py does — same logic.
# =========================
def recognition_loop():
    global latest_result
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame  = cv.flip(frame, 1)                          # mirror
        rgb    = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            with result_lock:
                latest_result = {'gesture': 'No hand detected',
                                 'confidence': 0, 'text': ''}
            continue

        num_hands = len(result.multi_hand_landmarks)

        if num_hands == 1:
            lm  = calc_landmark_list(frame, result.multi_hand_landmarks[0])
            pre = pre_process_landmark(lm)
            fv  = pre + PADDING                             # 84 features
        else:
            paired = list(zip(result.multi_hand_landmarks,
                              result.multi_handedness))
            paired.sort(key=lambda x: x[1].classification[0].label)
            fv = (pre_process_landmark(calc_landmark_list(frame, paired[0][0])) +
                  pre_process_landmark(calc_landmark_list(frame, paired[1][0])))

        sign_id    = keypoint_classifier(fv)
        gesture    = keypoint_classifier_labels[sign_id]
        confidence = get_confidence(fv, sign_id)

        with result_lock:
            latest_result = {'gesture':    gesture,
                             'confidence': confidence,
                             'text':       gesture}


# Start recognition in background thread
t = threading.Thread(target=recognition_loop, daemon=True)
t.start()


# =========================
# Video feed with landmarks
# =========================
# def generate_frames():
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         frame   = cv.flip(frame, 1)
#         rgb     = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#         results = hands.process(rgb)

#         if results.multi_hand_landmarks:
#             for hl in results.multi_hand_landmarks:
#                 mp.solutions.drawing_utils.draw_landmarks(
#                     frame, hl, mp_hands.HAND_CONNECTIONS)

#             # Show current prediction on frame
#             with result_lock:
#                 label = latest_result['gesture']
#             cv.putText(frame, label, (10, 50),
#                        cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

#         _, buf = cv.imencode('.jpg', frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' +
#                buf.tobytes() + b'\r\n')


# =========================
# Routes
# =========================
@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/gestures')
def get_gestures():
    return jsonify({'gestures': keypoint_classifier_labels})


@app.route('/api/config')
def get_config():
    return jsonify({'model': 'ISL KeyPoint Classifier',
                    'num_classes': len(keypoint_classifier_labels),
                    'labels': keypoint_classifier_labels})


@app.route('/api/result')
def get_result():
    """
    Frontend polls this every 800ms to get the latest gesture.
    No image sending needed — backend reads webcam directly.
    """
    with result_lock:
        return jsonify(latest_result)


# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# =========================
# Run
# =========================
if __name__ == '__main__':
    print("NeuroSign Backend Starting...")
    print(f"Classes: {keypoint_classifier_labels}")
    print("Running at http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
