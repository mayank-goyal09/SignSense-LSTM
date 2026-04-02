"""
Diagnostic script: Check data quality per action.
Reports how many frames have zero landmarks (no hand detected).
Also checks if the training data has enough variance between classes.
"""
import numpy as np
import os

DATA_PATH = 'MP_Data'
actions = ['hello', 'how_you', 'hi', 'whats_up', 'you_good']
no_sequences = 10
sequence_length = 30

print("=" * 60)
print("  SIGNSENSE DATA QUALITY REPORT")
print("=" * 60)

all_action_means = {}

for action in actions:
    zero_frames = 0
    total_frames = 0
    sequence_means = []
    
    for seq in range(no_sequences):
        seq_data = []
        for frame in range(sequence_length):
            path = os.path.join(DATA_PATH, action, str(seq), f"{frame}.npy")
            if os.path.exists(path):
                data = np.load(path)
                total_frames += 1
                if np.all(data == 0):
                    zero_frames += 1
                seq_data.append(data)
        
        if seq_data:
            seq_arr = np.array(seq_data)
            sequence_means.append(seq_arr.mean(axis=0))
    
    zero_pct = (zero_frames / total_frames * 100) if total_frames > 0 else 0
    
    # Average landmark vector for this action
    if sequence_means:
        action_mean = np.mean(sequence_means, axis=0)
        all_action_means[action] = action_mean
        variance = np.var(sequence_means, axis=0).mean()
    else:
        variance = 0
    
    print(f"\n📌 Action: {action.upper()}")
    print(f"   Total frames: {total_frames}")
    print(f"   Zero (no hand) frames: {zero_frames} ({zero_pct:.1f}%)")
    print(f"   Intra-class variance: {variance:.6f}")

# Cross-class similarity analysis
print("\n" + "=" * 60)
print("  CROSS-CLASS SIMILARITY (cosine similarity)")
print("=" * 60)
print(f"{'':>12}", end="")
for a in actions:
    print(f"{a:>12}", end="")
print()

for a1 in actions:
    print(f"{a1:>12}", end="")
    for a2 in actions:
        v1 = all_action_means[a1]
        v2 = all_action_means[a2]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(v1, v2) / (norm1 * norm2)
        else:
            sim = 0.0
        print(f"{sim:>12.4f}", end="")
    print()

print("\n" + "=" * 60)
print("  INTERPRETATION")
print("=" * 60)
print("- If zero-frame % is high: hand wasn't visible during collection")
print("- If cosine similarity between two actions is > 0.99: they look")
print("  almost IDENTICAL to the model and it can't tell them apart")
print("- If intra-class variance is very low: data is too uniform (overfitting risk)")
