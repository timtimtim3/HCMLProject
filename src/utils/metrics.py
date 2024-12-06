import torch

# Function to calculate metrics
def calculate_metrics(preds, labels, num_classes, device):
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
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    return macro_precision, macro_recall, macro_f1