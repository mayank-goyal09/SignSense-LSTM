
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import extract_landmarks # Import missing function

# Actions array MUST match training
actions = np.array(['hello', 'how_you', 'hi', 'whats_up', 'you_good'])

# 1. Initialize
model = load_model('action.h5')
sequence = []
sentence = []
threshold = 0.8 # Only show if we are >80% confident

# Use DirectShow for Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 2. Extract Landmarks (using our utility function)
    keypoints = extract_landmarks(frame)
    sequence.append(keypoints)
    sequence = sequence[-30:] # Keep only last 30 frames
    
    # 3. Prediction Logic
    if len(sequence) == 30:
        # Add verbose=0 to avoid terminal spam
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        
        # 4. Professional UI Logic
        action = actions[np.argmax(res)]
        confidence = res[np.argmax(res)]
        
        # Display logic: Show gesture with its confidence
        # Change color based on threshold (e.g. green if > 0.8, else orange)
        color = (0, 255, 0) if confidence > threshold else (0, 165, 255)
        
        # Simple UI overlay
        cv2.rectangle(frame, (0,0), (640, 40), (40, 40, 40), -1)
        cv2.putText(frame, f'GESTURE: {action.upper()} ({confidence*100:.1f}%)', (15,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('Real-time Sign Translator', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()