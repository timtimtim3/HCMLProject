import numpy as np
import argparse

def add_label_noise(y, noise_level, num_classes, random_seed=None):
    """
    Introduce label noise by randomly flipping labels.

    Parameters:
    - y: numpy array of true labels
    - noise_level: float, percentage of labels to corrupt (e.g., 0.1 for 10%)
    - num_classes: int, total number of classes
    - random_seed: int, random seed for reproducibility (optional)

    Returns:
    - y_noisy: numpy array with noisy labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_level * n_samples)

    # Randomly choose indices to corrupt
    noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)

    for idx in noisy_indices:
        true_label = y[idx]
        # Exclude the true label from possible new labels
        possible_labels = list(range(num_classes))
        possible_labels.remove(true_label)
        # Assign a random incorrect label
        new_label = np.random.choice(possible_labels)
        y_noisy[idx] = new_label

    return y_noisy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add label noise to y_train.npy")
    parser.add_argument('--input', type=str, default='y_train.npy', help='Input labels file (npy format)')
    parser.add_argument('--output', type=str, default='y_train_noisy.npy', help='Output labels file with noise (npy format)')
    parser.add_argument('--noise_level', type=float, default=0.1, help='Percentage of labels to corrupt (e.g., 0.1 for 10%)')
    parser.add_argument('--num_classes', type=int, default=49, help='Total number of classes')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Load true labels
    y_true = np.load(args.input)

    # Add label noise
    y_noisy = add_label_noise(y_true, args.noise_level, args.num_classes, args.random_seed)

    # Save noisy labels
    np.save(args.output, y_noisy)

    print(f"Saved noisy labels to {args.output}")
