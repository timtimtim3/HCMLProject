import numpy as np
import argparse
import json
import os
import matplotlib.pyplot as plt

from utils.parser import add_shared_parser_arguments
from utils.functions import get_output_dir_from_args, get_device, set_seed
from utils.logger import setup_logger

# Helper functions
def load_data(output_dir, dataset, label_noise, score_epoch_num):
    """
    Load the ground truth labels, noisy labels, and influence scores from the specified output directory.
    """

    # Load original labels
    y_true = np.load(os.path.join("data", dataset.upper(), f"labels_train.npy"))

    # Load noisy labels
    if label_noise != 0.0:
        y_noisy = np.load(os.path.join("data", dataset.upper(), f"labels_train_noisy{label_noise}.npy"))
    else:
        y_noisy = y_true.copy()

    # Load influence scores
    with open(os.path.join(output_dir, f"scores_epoch_{score_epoch_num}.json"), 'r') as f:
        influence_scores_ours = [entry['score'] for entry in json.load(f)]

    with open(os.path.join(output_dir, "baseline.json"), 'r') as f:
        influence_scores_baseline = [max(entry['probabilities']) for entry in json.load(f)]

    return np.array(y_true), np.array(y_noisy), np.array(influence_scores_ours), np.array(influence_scores_baseline)

def evaluate_noise_detection(noisy_indices, predicted_noisy_indices):
    """
    Evaluate the performance of noise detection.

    Args:
        noisy_indices (set): Indices of actual noisy labels.
        predicted_noisy_indices (set): Indices predicted as noisy.

    Returns:
        precision, recall, f1 (float): Evaluation metrics for noise detection.
    """
    true_positives = len(noisy_indices & predicted_noisy_indices)
    false_positives = len(predicted_noisy_indices - noisy_indices)
    false_negatives = len(noisy_indices - predicted_noisy_indices)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def detect_noisy_samples(influence_scores, top_percent, baseline=False):
    """
    Identify the top percent of samples as noisy based on influence scores.

    Args:
        influence_scores (np.ndarray): Influence scores for all samples.
        top_percent (float): The percentage of data to consider as noisy.
        baseline (bool): If True, sort in ascending order (for baseline logits).

    Returns:
        set: Indices of predicted noisy samples.
    """
    num_samples = len(influence_scores)
    num_noisy = int(top_percent * num_samples)
    ranked_indices = np.argsort(influence_scores if baseline else -influence_scores)  # Ascending for baseline, descending otherwise
    return set(ranked_indices[:num_noisy])

def plot_combined_results(args, f1_scores, output_path):
    """
    Plot combined F1 scores for resnet18 and resnet50 methods across different noise levels.

    Args:
        f1_scores (dict): Dictionary containing F1 scores for both models and methods.
        output_path (str): Path to save the plot.
    """
    label_noises = args.label_noises
    x = np.arange(len(label_noises))
    width = 0.2

    plt.figure(figsize=(14, 6))

    bars_resnet18_baseline = plt.bar(
        x - 0.5 * width, f1_scores['resnet18']['baseline'], width,
        color=(0.3922, 0.5843, 0.9294, 0.3), edgecolor="black", linewidth=2, label="Baseline (ResNet18)"
    )
    bars_resnet18_ours = plt.bar(
        x - 1.5 * width, f1_scores['resnet18']['ours'], width,
        color=(0.7686, 0.1176, 0.2275, 0.3), edgecolor="black", linewidth=2, label="Our Method (ResNet18)"
    )
    bars_resnet50_baseline = plt.bar(
        x + 1.5 * width, f1_scores['resnet50']['baseline'], width,
        color=(0.3922, 0.5843, 0.9294, 0.9), edgecolor="black", linewidth=2, label="Baseline (ResNet50)"
    )
    bars_resnet50_ours = plt.bar(
        x + 0.5 * width, f1_scores['resnet50']['ours'], width,
        color=(0.7686, 0.1176, 0.2275, 0.9), edgecolor="black", linewidth=2, label="Our Method (ResNet50)"
    )


    plt.xlabel("Noise Levels", fontsize=16)
    plt.ylabel("F1 Score", fontsize=16)
    plt.title("Noise Detection Performance on ISIC2024 using ResNet18 and ResNet50", fontsize=18)
    plt.xticks(x, [f"{int(n*100)}%" for n in label_noises], fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"Plot saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate noise detection performance for multiple models.")
    add_shared_parser_arguments(parser)
    parser.add_argument("--label_noises", nargs='+', required=True, type=float, help="Specify the label noises used during training")
    parser.add_argument("--epochs_resnet18", nargs='+', required=True, type=int, help="Epoch numbers for ResNet18")
    parser.add_argument("--epochs_resnet50", nargs='+', required=True, type=int, help="Epoch numbers for ResNet50")

    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger()
    device = get_device()

    # Log parameters
    logger.info(f'Using device: {device}')
    logger.info(f'Args: {vars(args)}')

    models = ['resnet18', 'resnet50']
    f1_scores = {model: {'ours': [], 'baseline': []} for model in models}

    for model in models:
        args.model = model
        epochs = args.epochs_resnet18 if model == 'resnet18' else args.epochs_resnet50

        for index, label_noise in enumerate(args.label_noises):
            args.label_noise = label_noise
            output_dir = get_output_dir_from_args(args)

            # Load data
            y_true, y_noisy, influence_scores_ours, influence_scores_baseline = load_data(output_dir, args.dataset, label_noise, epochs[index])

            # Identify actual noisy samples
            noisy_indices = set(np.where(y_true != y_noisy)[0])

            # Predict noisy samples using our method and baseline method
            predicted_noisy_ours = detect_noisy_samples(influence_scores_ours, label_noise)
            predicted_noisy_baseline = detect_noisy_samples(influence_scores_baseline, label_noise, baseline=True)

            # Evaluate both methods
            _, _, f1_ours = evaluate_noise_detection(noisy_indices, predicted_noisy_ours)
            _, _, f1_baseline = evaluate_noise_detection(noisy_indices, predicted_noisy_baseline)

            f1_scores[model]['baseline'].append(f1_baseline)
            f1_scores[model]['ours'].append(f1_ours)
            

    # Plot combined results
    plot_path = os.path.join(get_output_dir_from_args(args), f"../combined_noise_detection_performance_{args.dataset}.png")
    plot_combined_results(args, f1_scores, plot_path)

if __name__ == "__main__":
    main()
