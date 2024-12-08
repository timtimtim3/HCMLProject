import torch
import numpy as np
import random


def set_seed(seed):
    """Set the seed for torch, random and numpy for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the torch device, this can be cuda or cpu."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_checkpoint_dir_from_args(args):
    """This folder contains all the checkpoints."""

    return f"checkpoints/{args.dataset}_{args.model}_"\
        f"{args.hidden_sizes}_{args.lr}_{args.batch_size}_{args.num_epochs}_{args.seed}"


def calculate_metrics(preds, labels, num_classes, device):
    """
    Calculate classification metrics including accuracy, macro-averaged precision, recall, and F1 score.

    Args:
        preds (torch.Tensor): Predicted class labels (1D tensor).
        labels (torch.Tensor): True class labels (1D tensor).
        num_classes (int): Number of classes in the dataset.
        device (torch.device): Device where tensors are allocated (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the following metrics:
            - accuracy (float): Overall accuracy of the predictions.
            - macro_precision (float): Macro-averaged precision across all classes.
            - macro_recall (float): Macro-averaged recall across all classes.
            - macro_f1 (float): Macro-averaged F1 score across all classes.
    """
    
    # Initialize variables to store true positives, false positives, false negatives
    true_positives = torch.zeros(num_classes, device=device)
    false_positives = torch.zeros(num_classes, device=device)
    false_negatives = torch.zeros(num_classes, device=device)

    for cls in range(num_classes):
        # For each class, calculate true positives, false positives, false negatives
        true_positives[cls] = ((preds == cls) & (labels == cls)).sum()
        false_positives[cls] = ((preds == cls) & (labels != cls)).sum()
        false_negatives[cls] = ((preds != cls) & (labels == cls)).sum()

    # Calculate precision, recall for each class
    precision = torch.zeros(num_classes, device=device)
    recall = torch.zeros(num_classes, device=device)
    f1 = torch.zeros(num_classes, device=device)

    for cls in range(num_classes):
        if true_positives[cls] + false_positives[cls] > 0:
            precision[cls] = true_positives[cls] / (true_positives[cls] + false_positives[cls])
        else:
            precision[cls] = 0.0
        if true_positives[cls] + false_negatives[cls] > 0:
            recall[cls] = true_positives[cls] / (true_positives[cls] + false_negatives[cls])
        else:
            recall[cls] = 0.0
        if precision[cls] + recall[cls] > 0:
            f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
        else:
            f1[cls] = 0.0

    # Compute macro-average

    accuracy = (preds == labels).sum().item() / len(labels)
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    return accuracy, macro_precision, macro_recall, macro_f1


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


def map_isic_label_to_binary(label_str):
    """
    Map the label string to binary:
    - "benign" or "indeterminate/benign" -> 0
    - "malignant" or "indeterminate/malignant" -> 1
    - "indeterminate" or "unlabeled" -> None (exclude these)
    """
    if not isinstance(label_str, str):
        return None
    label_str = label_str.lower().strip()

    if "benign" in label_str:
        return 0
    elif "malignant" in label_str:
        return 1
    else:
        # For unlabeled, indeterminate, etc.
        return None