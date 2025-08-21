# Sign Language Translator

A fullstack web application that translates sign language to text using computer vision and machine learning.


https://github.com/user-attachments/assets/de1f11cd-d656-43e7-9e70-cbcc640aa531


## Features

- Real-time webcam capture
- Sign language recognition using MediaPipe
- Modern React frontend with Tailwind CSS
- Flask backend for processing

## Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/MacOS
```

3. Install dependencies:
```bash
pip install flask flask-cors mediapipe opencv-python numpy
```

4. Run the Flask server:
```bash
python app.py
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

## Usage

1. Open your browser and navigate to `http://localhost:5173`
2. Allow camera access when prompted
3. Click "Start Capturing" to begin sign language recognition
4. Perform sign language gestures in front of the camera
5. The translation will appear below the video feed

## Technologies Used

- Frontend: React, Vite, Tailwind CSS
- Backend: Python, Flask
- Computer Vision: MediaPipe, OpenCV
- API Communication: Axios
