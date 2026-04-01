
import cv2
import numpy as np
import os
import sys
import mediapipe as mp

# ==========================================
# PART 1: SETUP MEDIA PIPE (Robust & Standalone)
# ==========================================
print("Initializing MediaPipe...")
USING_TASKS_API = False
detector = None
hands = None

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Check for model file in current directory
    MODEL_PATH = 'hand_landmarker.task'
    
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
            sys.exit(1)
    
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    print("[INFO] Using Legacy MediaPipe API")

def extract_landmarks(frame):
    if USING_TASKS_API:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        if detection_result.hand_landmarks:
            return np.array([[res.x, res.y, res.z] for res in detection_result.hand_landmarks[0]]).flatten()
    else:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            return np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten()
    return np.zeros(63)

# ==========================================
# PART 2: DATA COLLECTION CONFIGURATION
# ==========================================
DATA_PATH = os.path.join('MP_Data') 
ACTIONS = np.array(['hello', 'thanks', 'iloveyou'])
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30

# ==========================================
# PART 3: MAIN LOOP
# ==========================================
def main():
    # 1. Setup Camera (DirectShow for Windows stability)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 2. Setup Directories
    for action in ACTIONS: 
        for sequence in range(NO_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

    print("\n=================================================")
    print(f"COLLECTING DATA FOR: {ACTIONS}")
    print("Instructions:")
    print("1. A preview window will open.")
    print("2. Press 'S' to START the collection.")
    print("3. Press 'Q' at any time to QUIT.")
    print("=================================================\n")

    # 3. Preview Loop (Wait for Start)
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Visual Feedback: Detection Check
        results = extract_landmarks(frame)
        if np.any(results):
            # Draw landmarks
            reshaped = results.reshape(21, 3)
            h, w, c = frame.shape
            for point in reshaped:
                cx, cy = int(point[0] * w), int(point[1] * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, "HAND DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO HAND DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, 'PRESS "S" TO START', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('OpenCV Feed', frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            break
        if key & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # 4. Collection Loop
    for action in ACTIONS:
        # Give user a break between actions
        print(f"Preparing for action: {action}")
        for i in range(50): # Short pause
            ret, frame = cap.read()
            cv2.putText(frame, f'Get Ready for {action}!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('OpenCV Feed', frame)
            cv2.waitKey(20)

        for sequence in range(NO_SEQUENCES):
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret: break

                # Extract
                results = extract_landmarks(frame)
                
                # Save
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, results)

                # Visualize
                if np.any(results):
                    reshaped = results.reshape(21, 3)
                    h, w, c = frame.shape
                    for point in reshaped:
                        cx, cy = int(point[0] * w), int(point[1] * h)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                cv2.putText(frame, f'Collection: {action} | Video: {sequence} | Frame: {frame_num}', 
                           (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete.")

if __name__ == "__main__":
    main()
