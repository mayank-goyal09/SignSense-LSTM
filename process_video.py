
import cv2
import os
import numpy as np
from lm_utils import extract_landmarks

# Define the segments (Start frame, End frame) assuming 30fps
segments = {
    "hello": (0, 45),
    "how_you": (60, 105),
    "hi": (120, 165),
    "whats_up": (180, 225),
    "you_good": (240, 315)
}

video_path = 'downloaded_video.mp4'
DATA_PATH = 'MP_Data'

if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    print("Please make sure you have the video file in this folder.")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

print(f"Processing {video_path}...")

for action, (start_frame, end_frame) in segments.items():
    print(f"Extracting '{action}' from frame {start_frame} to {end_frame}...")
    
    # We will repeat the segment multiple times to create "sequences"
    # Since it's the exact same frames, this is just data duplication 
    # but it fits the training structure.
    for sequence in range(10): 
        # Set video to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        for frame_num in range(30):
            ret, frame = cap.read()
            if not ret or current_frame > end_frame: 
                # If segment is shorter than 30 frames, padding might be needed
                # Here we just break, but for robust training fixed size is better
                break
            
            # Extract the 'Skeletal' data
            landmarks = extract_landmarks(frame)
            
            # Save as .npy
            path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, landmarks)
            
            current_frame += 1
            
    print(f"✅ Processed and saved: {action}")

cap.release()
print("All segments processed.")
