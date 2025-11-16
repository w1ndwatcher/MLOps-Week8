# 35.232.213.140
import numpy as np
import pandas as pd
import random

def inject_label_noise(labels: pd.Series, noise_ratio: float) -> pd.Series:
    """
    Introduce label noise by randomly flipping a percentage of entries.

    Args:
        labels: Series containing the original class labels.
        noise_ratio: Fraction (0–1) of samples whose labels should be flipped.

    Returns:
        A new Series with noisy (flipped) labels.
    """
    # No poisoning requested → return original unchanged
    if noise_ratio <= 0:
        return labels.copy()

    classes = labels.unique()

    # If classification has only one class, flipping is impossible
    if len(classes) < 2:
        return labels.copy()

    noisy_labels = labels.copy()
    total = len(noisy_labels)
    count_to_flip = int(total * noise_ratio)

    # If too small ratio yields zero flips
    if count_to_flip == 0:
        print(f"Note: noise_ratio={noise_ratio} resulted in zero label flips.")
        return noisy_labels

    # Randomly choose which indices will be modified
    selected_idx = np.random.choice(noisy_labels.index, size=count_to_flip, replace=False)

    print(f"Injecting noise into {count_to_flip}/{total} labels...")

    for idx in selected_idx:
        original = noisy_labels.at[idx]

        # Select any class except the original one
        candidates = [c for c in classes if c != original]
        replacement = random.choice(candidates)

        noisy_labels.at[idx] = replacement

    return noisy_labels