from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import os

print("Starting training script...")

DATA_PATH = os.path.join('MP_Data')
# Actions we are predicting
actions = np.array(['hello', 'how_you', 'hi', 'whats_up', 'you_good'])
no_sequences = 30 # From process_video loop range(30)? No, loop was range(10) in process_video.py but inner loop was 30 frames.
# Wait, process_video.py had: for sequence in range(10): for frame_num in range(30):
# So valid sequences are 0-9 (10 total).
# Let's verify inspecting the folders. The list_dir showed 300 children for 'hello'? 
# If 10 sequences * 30 frames = 300 files. Yes. 
# So no_sequences should be 10.
no_sequences = 10
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

print("Loading data...")
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        try:
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
        except FileNotFoundError:
            print(f"Warning: Missing data for {action} sequence {sequence}. Skipping.")

print(f"Data loaded. Found {len(sequences)} sequences.")
if len(sequences) == 0:
    print("Error: No valid sequences found. Please run process_video.py first.")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Use stratify to ensure all classes are represented in test set
# Dataset is small (50 samples), so we need a larger test split (e.g. 20% = 10 samples) to fit all 5 classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential([
    # Layer 1: Takes (30 frames, 63 landmarks)
    # Using Input(shape) is cleaner as per warning
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    # Layer 2: Deeper features
    LSTM(128, return_sequences=True, activation='relu'),
    # Layer 3: Final sequence processing
    LSTM(64, return_sequences=False, activation='relu'),
    
    # Fully Connected Layers
    Dense(64, activation='relu'),
    BatchNormalization(), # Keeps training stable
    Dropout(0.2), # Prevents overfitting (crucial for small datasets!)
    
    # Output Layer: Softmax gives us a probability for each of the 5 signs
    Dense(actions.shape[0], activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Starting training...")
# 2000 epochs is likely overkill for 50 samples, reducing to 200-500 or keeping high with early stopping intent
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

print("Saving model...")
model.save('action.h5')
print("Model saved as action.h5")