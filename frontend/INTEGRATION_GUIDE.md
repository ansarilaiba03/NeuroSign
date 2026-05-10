# NeuroSign - Frontend & Backend Integration Guide

## Overview
NeuroSign is a sign language gesture recognition system that converts hand gestures to text in real-time using AI and computer vision.

## Current Setup

### Backend (Flask API)
- **Server**: Running at `http://localhost:5000`
- **Location**: `backend/server.py`
- **Dependencies**: OpenCV, MediaPipe, Flask, Flask-CORS
- **Status**: ✅ Running and connected

### Frontend (HTML/CSS/JavaScript)
- **Location**: `index.html`, `styles.css`, `script.js`
- **Connection**: Integrated with backend API
- **Fallback**: Demo mode if backend unavailable

## How to Use

### 1. Start the Backend
```bash
cd backend
.\venv\Scripts\Activate.ps1
python server.py
```
The server will start at `http://localhost:5000`

### 2. Open Frontend
Open `index.html` in your browser or serve it via a local web server:
```bash
# Using Python
python -m http.server 8000

# Then navigate to http://localhost:8000
```

### 3. Use Live Recognition
1. Click "Start Camera" button
2. Allow browser camera access
3. Perform hand gestures in front of the camera
4. Gestures are recognized and displayed in real-time
5. Text output is shown below the confidence bar
6. Click "Speak Text" to hear the recognized text

## API Endpoints

### Health Check
```
GET /health
```
Response: `{ "status": "ok", "message": "NeuroSign API is running" }`

### Gesture Recognition
```
POST /api/recognize
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,..."
}
```
Response:
```json
{
  "gesture": "Hello",
  "confidence": 92,
  "num_hands": 1,
  "landmarks": [...],
  "processed_image": "data:image/jpeg;base64,..."
}
```

### Available Gestures
```
GET /api/gestures
```
Returns list of all recognizable gestures

### Configuration
```
GET /api/config
```
Returns API configuration and available endpoints

## Features

✅ Real-time hand gesture recognition
✅ Confidence scoring
✅ Gesture history tracking
✅ Text-to-speech conversion
✅ Responsive web design
✅ Demo mode fallback
✅ CORS enabled for cross-origin requests

## Browser Compatibility

- Chrome/Chromium: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- Edge: ✅ Full support

**Note**: Requires HTTPS for camera access in production (HTTP OK for localhost)

## Troubleshooting

### Backend not connecting?
- Ensure Flask server is running: `python server.py`
- Check if port 5000 is available
- Verify backend console shows "Running on http://127.0.0.1:5000"

### Camera not working?
- Check browser camera permissions
- Ensure HTTPS (or localhost)
- Try a different browser
- Check browser console for errors

### No gestures detected?
- Ensure adequate lighting
- Make sure hands are clearly visible
- Hold gestures steady for 2-3 seconds
- App will show "No hand detected" if hands not visible

## Project Structure

```
neurosign/
├── index.html          # Main webpage
├── styles.css          # Styling
├── script.js           # Frontend logic (integrated with API)
├── backend/
│   ├── server.py       # Flask API server
│   ├── app.py          # Original gesture recognition app
│   ├── model/          # Pre-trained ML models
│   ├── utils/          # Utility functions
│   ├── requirements_run.txt
│   └── venv/           # Python virtual environment
└── README.md
```

## Development Notes

- Backend API is designed to be stateless
- Each frame is sent independently for recognition
- Confidence scores are based on model predictions
- Landmark data includes x, y, z coordinates for hand joints
- Processing happens server-side for better accuracy

## Future Enhancements

- [ ] Integration of full ML model for better accuracy
- [ ] Real-time video streaming instead of frame sampling
- [ ] Support for more complex gestures
- [ ] Training interface for custom gestures
- [ ] Database for gesture history
- [ ] Multi-language support
