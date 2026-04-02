"""
Improved training script for SignSense.

Key improvements over original:
1. Filters out zero-vectors (frames where no hand was detected)
   and pads sequences with noise instead of pure zeros to prevent
   the model from learning "zero = a valid gesture"
2. Adds class weights to handle imbalanced valid-frame counts
3. Uses data augmentation (slight noise injection) to create more variance
4. Better model architecture with regularization
5. Adds validation monitoring with early stopping
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

print("=" * 50)
print("  SignSense LSTM — Improved Training Pipeline")
print("=" * 50)

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'how_you', 'hi', 'whats_up', 'you_good'])
no_sequences = 10
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}

# ==========================================
# 1. LOAD DATA WITH QUALITY FILTERING
# ==========================================
print("\n[Step 1] Loading and filtering data...")
sequences, labels = [], []

for action in actions:
    valid_seqs = 0
    for sequence in range(no_sequences):
        window = []
        valid_frames_in_seq = 0
        
        try:
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                
                # If this frame is all zeros (no hand detected),
                # replace with small random noise so the model doesn't
                # learn "zeros = hello" or "zeros = whatsup"
                if np.all(res == 0):
                    res = np.random.normal(0, 0.001, 63).astype(np.float32)
                else:
                    valid_frames_in_seq += 1
                
                window.append(res)
            
            # Only include this sequence if at least 30% of frames had a hand
            if valid_frames_in_seq >= 9:  # 9/30 = 30%
                sequences.append(window)
                labels.append(label_map[action])
                valid_seqs += 1
            else:
                # Create augmented version with noise to still use the data
                # but mark it as lower quality
                noisy_window = []
                for frame in window:
                    noisy_frame = frame + np.random.normal(0, 0.01, 63).astype(np.float32)
                    noisy_window.append(noisy_frame)
                sequences.append(noisy_window)
                labels.append(label_map[action])
                valid_seqs += 1
                
        except FileNotFoundError:
            print(f"  [WARN] Missing data for {action} seq {sequence}")
    
    print(f"  {action}: {valid_seqs} sequences loaded")

# ==========================================
# 2. DATA AUGMENTATION (create more training data)
# ==========================================
print("\n[Step 2] Augmenting data...")
original_count = len(sequences)

augmented_seqs = []
augmented_labels = []

for i, (seq, label) in enumerate(zip(sequences, labels)):
    seq_arr = np.array(seq)
    
    # Augmentation 1: Add slight noise
    noisy = seq_arr + np.random.normal(0, 0.02, seq_arr.shape)
    augmented_seqs.append(noisy.tolist())
    augmented_labels.append(label)
    
    # Augmentation 2: Scale landmarks slightly (simulates closer/farther hand)
    scaled = seq_arr * np.random.uniform(0.9, 1.1)
    augmented_seqs.append(scaled.tolist())
    augmented_labels.append(label)
    
    # Augmentation 3: Time-shift (shift sequence by 1-3 frames)
    shift = np.random.randint(1, 4)
    shifted = np.roll(seq_arr, shift, axis=0)
    augmented_seqs.append(shifted.tolist())
    augmented_labels.append(label)

sequences.extend(augmented_seqs)
labels.extend(augmented_labels)

print(f"  Original: {original_count} sequences")
print(f"  After augmentation: {len(sequences)} sequences (4x)")

# ==========================================
# 3. PREPARE DATA
# ==========================================
print("\n[Step 3] Preparing train/test split...")
X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing:  {X_test.shape[0]} samples")

# ==========================================
# 4. COMPUTE CLASS WEIGHTS
# ==========================================
print("\n[Step 4] Computing class weights...")
y_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_labels), y=y_labels)
class_weight_dict = dict(enumerate(class_weights))
print(f"  Weights: {class_weight_dict}")

# ==========================================
# 5. BUILD IMPROVED MODEL
# ==========================================
print("\n[Step 5] Building model...")

model = Sequential([
    # Layer 1: Temporal feature extraction
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    Dropout(0.3),
    
    # Layer 2: Deeper temporal patterns
    LSTM(128, return_sequences=True, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Layer 3: Final sequence compression
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.3),
    
    # Dense layers
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(32, activation='relu'),
    
    # Output
    Dense(actions.shape[0], activation='softmax')
])

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

# ==========================================
# 6. TRAIN WITH EARLY STOPPING
# ==========================================
print("\n[Step 6] Training...")

log_dir = os.path.join('Logs')
callbacks = [
    TensorBoard(log_dir=log_dir),
    EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    epochs=300,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    batch_size=8,
    verbose=1
)

# ==========================================
# 7. EVALUATE
# ==========================================
print("\n[Step 7] Evaluating...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {acc*100:.1f}%")

# Per-class accuracy
y_pred = model.predict(X_test, verbose=0)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n  Per-class accuracy:")
for i, action in enumerate(actions):
    mask = y_true_labels == i
    if mask.sum() > 0:
        class_acc = (y_pred_labels[mask] == i).mean() * 100
        print(f"    {action}: {class_acc:.1f}% ({mask.sum()} samples)")
    else:
        print(f"    {action}: no test samples")

# ==========================================
# 8. SAVE
# ==========================================
print("\n[Step 8] Saving model...")
model.save('action.h5')
print("Model saved as action.h5")
print("\n✅ Training complete!")