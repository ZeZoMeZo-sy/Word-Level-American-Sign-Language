import pandas as pd
import numpy as np


# ====================== BALANCING ======================

def get_balanced_df(csv_path, target_count=60):
    """
    Ensures every class has at least target_count training samples
    by oversampling minority classes.
    """
    df       = pd.read_csv(csv_path)
    train_df = df[df['split'] == 'train'].copy()

    balanced_list = []

    for label in train_df['label'].unique():
        class_subset  = train_df[train_df['label'] == label]
        current_count = len(class_subset)

        if current_count >= target_count:
            # Cap majority classes so they don't dominate
            balanced_list.append(class_subset.sample(n=target_count, replace=False))
        else:
            extra_needed  = target_count - current_count
            extra_samples = class_subset.sample(n=extra_needed, replace=True)
            balanced_list.append(pd.concat([class_subset, extra_samples]))

    return pd.concat(balanced_list).reset_index(drop=True)


# ====================== AUGMENTATION ======================

def augment_landmarks(data):
    """
    Keypoint-aware augmentation pipeline.

    data shape: (T, F) where F = 150
        - Pose:       indices 0..65   → 33 landmarks × (x, y)  = 66 values
        - Left hand:  indices 66..107 → 21 landmarks × (x, y)  = 42 values
        - Right hand: indices 108..149→ 21 landmarks × (x, y)  = 42 values

    All augmentations operate on the RAW coordinates BEFORE velocity is
    computed, so velocity stays consistent with the transformed positions.
    """

    data = data.copy()   # never mutate the original

    # ----------------------------------------------------------
    # 1. SPATIAL SCALING
    #    Simulates signer being closer/further from camera.
    #    Applied uniformly to all x and y so relative positions hold.
    # ----------------------------------------------------------
    scale = np.random.uniform(0.85, 1.15)
    data  = data * scale

    # ----------------------------------------------------------
    # 2. SPATIAL SHIFT  (FIX: stride=2 for (x,y) layout)
    #    Simulates signer standing left/right/up/down in frame.
    #    Pose:  every even index = x, every odd index = y
    #    Hands: same (x,y) pairs
    # ----------------------------------------------------------
    shift_x = np.random.uniform(-0.05, 0.05)
    shift_y = np.random.uniform(-0.05, 0.05)

    # Pose x-coords (0, 2, 4, ... 64) and y-coords (1, 3, 5, ... 65)
    data[:, 0:66:2]   += shift_x
    data[:, 1:66:2]   += shift_y

    # Left hand x-coords (66, 68, ... 106) and y-coords (67, 69, ... 107)
    data[:, 66:108:2]  += shift_x
    data[:, 67:108:2]  += shift_y

    # Right hand x-coords (108, 110, ... 148) and y-coords (109, 111, ... 149)
    data[:, 108:150:2] += shift_x
    data[:, 109:150:2] += shift_y

    # ----------------------------------------------------------
    # 3. HORIZONTAL FLIP  (50% chance)
    #    Mirrors the sign left-right. Critical for WLASL because
    #    many signs are one-handed and signers vary dominant hand.
    #    Flip = negate all x-coords, then swap left/right hand blocks.
    # ----------------------------------------------------------
    if np.random.random() < 0.5:
        # Negate all x-coords
        data[:, 0:66:2]   *= -1
        data[:, 66:108:2]  *= -1
        data[:, 108:150:2] *= -1

        # Swap left hand ↔ right hand blocks
        left_hand  = data[:, 66:108].copy()
        right_hand = data[:, 108:150].copy()
        data[:, 66:108]  = right_hand
        data[:, 108:150] = left_hand

    # ----------------------------------------------------------
    # 4. TEMPORAL WARPING
    #    Speeds up or slows down the signing motion.
    #    Forces model to be invariant to signing speed.
    # ----------------------------------------------------------
    T = data.shape[0]
    if np.random.random() < 0.6:
        warp_factor = np.random.uniform(0.8, 1.2)
        old_indices = np.linspace(0, T - 1, T)
        new_indices = np.linspace(0, T - 1, int(T * warp_factor))
        new_indices = np.clip(new_indices, 0, T - 1)

        # Resample each feature along time axis
        warped = np.zeros_like(data)
        for f in range(data.shape[1]):
            warped[:, f] = np.interp(old_indices, new_indices, 
                                     np.interp(new_indices, old_indices, data[:, f]))
        data = warped

    # ----------------------------------------------------------
    # 5. TEMPORAL MASKING
    #    Zeroes out 1-3 random frames.
    #    Simulates MediaPipe detection dropout.
    # ----------------------------------------------------------
    if np.random.random() < 0.5:
        num_mask    = np.random.randint(1, 4)
        mask_frames = np.random.choice(T, size=num_mask, replace=False)
        data[mask_frames, :] = 0

    # ----------------------------------------------------------
    # 6. GAUSSIAN NOISE
    #    Very subtle coordinate jitter. Applied last so it doesn't
    #    interact with the structured augmentations above.
    # ----------------------------------------------------------
    noise = np.random.normal(0, 0.003, data.shape)
    data  = data + noise

    return data