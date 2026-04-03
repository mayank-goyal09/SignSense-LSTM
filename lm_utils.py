
import cv2
import numpy as np
import os
import sys
import mediapipe as mp

# ==========================================
# MEDIA PIPE SETUP (Robust)
# ==========================================
print("Initializing MediaPipe in utils...")
USING_TASKS_API = False
detector = None
hands = None

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Check for model file in current directory or same directory as utils.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(current_dir, 'hand_landmarker.task')
    
    if os.path.exists(MODEL_PATH):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        detector = vision.HandLandmarker.create_from_options(options)
        USING_TASKS_API = True
        print("[INFO] Using MediaPipe Tasks API")
    else:
        print(f"[WARNING] '{MODEL_PATH}' not found. Will attempt legacy API fallback.")
except ImportError:
    pass

if not USING_TASKS_API:
    try:
        mp_hands = mp.solutions.hands
    except AttributeError:
        try:
            import mediapipe.python.solutions as solutions
            mp_hands = solutions.hands
        except ImportError:
            print("[CRITICAL] Could not load MediaPipe. Please ensure it is installed.")
    
    if 'mp_hands' in locals():
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        print("[INFO] Using Legacy MediaPipe API")

def extract_landmarks(frame):
    if USING_TASKS_API and detector:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        if detection_result.hand_landmarks:
            return np.array([[res.x, res.y, res.z] for res in detection_result.hand_landmarks[0]]).flatten()
    elif hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            return np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
            
    return np.zeros(63)
