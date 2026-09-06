# NeuroSign 🤟

An AI-based system that converts sign language gestures into text (and speech) using computer vision and machine learning.

## 🌟 Features

- **Real-time gesture recognition** via webcam input
- **Keypoint-based classification** — hand landmarks are extracted and classified using a trained model
- **Text-to-speech output** — recognized signs are converted to spoken audio
- **Gesture history tracking**
- **Responsive demo web interface** showcasing the recognition pipeline: Input → Preprocessing → Feature Extraction → Classification → Text Output → Speech Output

## 🔤 Dataset — Indian Sign Language (ISL)

NeuroSign recognizes the **Indian Sign Language (ISL) alphabet** (A–Z), where each letter is represented by a distinct static hand pose (with a couple of letters like H, J, and Z involving a small motion). The chart below is the reference used for building the dataset:

![Indian Sign Language Alphabet Chart](./assets/isl_alphabet_chart.jpg)

Since ISL letters differ from ASL in several shapes (e.g. many ISL letters are formed using **two hands**, unlike ASL's one-handed alphabet), a custom dataset was created rather than reusing an existing ASL dataset.

### How the dataset was built

The model doesn't classify raw images — it classifies **hand landmark keypoints** extracted by [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker), which is lighter, faster, and more robust to background/lighting than raw image classification. This matches the `keypoint_classification.ipynb` and `point_history_classification.ipynb` notebooks in `backend/`.

**Pipeline:**
1. **Capture landmarks, not pixels** — For each webcam frame, MediaPipe Hands detects 21 keypoints per hand (x, y, z coordinates of each knuckle/fingertip).
2. **Normalize** — Coordinates are converted to relative positions (relative to the wrist) and scaled, so the same gesture looks the same regardless of hand size or distance from the camera.
3. **Label and log** — While holding a specific letter's hand shape, the normalized landmark vector is appended as a row to a CSV file, tagged with the letter's class label (e.g. `0` for A, `1` for B, ...).
4. **Repeat per class** — This is done for all 26 letters, capturing many samples (ideally 100s per letter, from different angles/hand sizes/lighting) to build a balanced dataset.
5. **Train a classifier** — The CSV of `(landmark_vector, label)` pairs is used to train a lightweight classifier (e.g. a small neural network or `RandomForest`/`SVM`), producing the model used in `backend/model/`.
6. **Point-history model (for motion letters)** — For letters that involve movement (like H, J, Z), a second dataset tracks the *trajectory* of a keypoint (e.g. fingertip) over several frames, and a separate point-history classifier is trained to recognize the motion pattern.

### 🧑‍🏫 How to create your own dataset (e.g. for a different sign language or gesture set)

1. **Set up MediaPipe Hands** to run on your webcam feed and extract 21 landmark points per detected hand each frame.
2. **Pick your classes** — decide the full list of gestures/letters you want to recognize and assign each a numeric label.
3. **Build a data-collection script** — capture keyboard input (e.g. pressing `0`–`9` or a letter key) to mark "I am currently showing gesture X," and on each frame while that key is held, save the normalized landmark vector + label to a CSV (`keypoint.csv`).
4. **Normalize consistently** — always convert landmarks to be relative to a fixed reference point (like the wrist) and scale them, so the dataset isn't sensitive to hand position/size in the frame.
5. **Collect enough samples per class** — aim for a few hundred samples per gesture, varied across different people, hand sizes, angles, and lighting conditions to avoid overfitting.
6. **Split into train/validation sets** — e.g. an 80/20 split, so you can measure real accuracy.
7. **Train a classifier** on the CSV (a small feedforward neural network works well for this size of input, ~42-63 features per sample depending on 1 or 2 hands).
8. **Export/save the trained model** and plug it into the inference script (`backend/app.py`) in place of the existing model.
9. **(Optional) Repeat steps 3–8 with a point-history CSV** if any of your gestures involve motion rather than a static pose.

This approach (MediaPipe landmarks + a small classifier) makes it easy to add new gestures without retraining a full deep learning image model — you just collect more labeled landmark rows and retrain the lightweight classifier.

## 🎯 Applications

- Communication aid for the deaf/hard-of-hearing community
- Educational learning tool for sign language
- Human-computer interaction and accessibility technology

## 🛠️ Tech Stack

**Backend**
- Python, Flask/FastAPI-style `app.py` server
- Keypoint & point-history classification models (trained via Jupyter notebooks)
- Separate `requirements_train.txt` and `requirements_run.txt` for training vs. inference environments

**Frontend**
- HTML, CSS, JavaScript
- Web Speech API for text-to-speech

## 📂 Structure

```
NeuroSign/
├── backend/     # Model, inference server, training notebooks
└── frontend/    # Demo web interface
```

## 🚀 Getting Started

**Backend:**
```bash
cd backend
pip install -r requirements_run.txt
python app.py
```

**Frontend:**
Open `frontend/index.html` in a browser (see `frontend/INTEGRATION_GUIDE.md` for connecting it to the backend).

## 📌 Notes

This was built as a mini-project demonstrating an end-to-end gesture-recognition pipeline. See `frontend/DEMO_GUIDE.md` for a walkthrough of the live demo.
