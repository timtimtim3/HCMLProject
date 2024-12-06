

def metrics():
    pass


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