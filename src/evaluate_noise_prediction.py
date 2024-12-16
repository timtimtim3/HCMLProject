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


def plot_results(args, f1_scores_ours, f1_scores_baseline, output_path):
    """
    Plot F1 scores for our method and baseline method across different noise levels.

    Args:
        label_noises (list): List of noise levels.
        f1_scores_ours (list): F1 scores for our method.
        f1_scores_baseline (list): F1 scores for the baseline method.
        output_path (str): Path to save the plot.
    """
    model_name = args.model
    dataset_name = args.dataset
    label_noises = args.label_noises


    x = np.arange(len(label_noises))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, f1_scores_ours, width, color="#8B0000", label="Our Method")  # Bordeaux red
    plt.bar(x + width/2, f1_scores_baseline, width, color="#008B8B", label="Baseline")  # Dark aqua blue

    plt.xlabel("Noise Levels", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.title(f"Noise Detection Performance ({dataset_name}, {model_name})", fontsize=16)
    plt.xticks(x, [f"{int(n*100)}%" for n in label_noises], fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"Plot saved at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate noise detection performance.")
    add_shared_parser_arguments(parser)
    parser.add_argument("--label_noises", nargs='+', required=True, type=float, help="Specify the label noises used during training")
    parser.add_argument("--epochs", nargs="+", type=int, required=True, help="Specify a list of epoch numbers for which the scores are loaded")

    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger() 
    device = get_device() 

    # Log parameters
    logger.info(f'Using device: {device}')
    logger.info(f'Args: {vars(args)}')

    f1_scores_ours = []
    f1_scores_baseline = []

    for index, label_noise in enumerate(args.label_noises):

        args.label_noise = label_noise
        output_dir = get_output_dir_from_args(args)

        # Load data
        y_true, y_noisy, influence_scores_ours, influence_scores_baseline = load_data(output_dir, args.dataset, label_noise, args.epochs[index])

        # Identify actual noisy samples
        noisy_indices = set(np.where(y_true != y_noisy)[0])
        total_noisy = len(noisy_indices)

        print(f"Noise Level: {label_noise}, Total samples: {len(y_true)}, Total noisy samples: {total_noisy}")

        # Predict noisy samples using our method and baseline method
        predicted_noisy_ours = detect_noisy_samples(influence_scores_ours, args.label_noise)
        predicted_noisy_baseline = detect_noisy_samples(influence_scores_baseline, args.label_noise, baseline=True)

        # Evaluate both methods
        _, _, f1_ours = evaluate_noise_detection(noisy_indices, predicted_noisy_ours)
        _, _, f1_baseline = evaluate_noise_detection(noisy_indices, predicted_noisy_baseline)

        f1_scores_ours.append(f1_ours)
        f1_scores_baseline.append(f1_baseline)

        print(f"F1 Score (Our Method): {f1_ours:.4f}, F1 Score (Baseline): {f1_baseline:.4f}")

    plot_path = os.path.join(output_dir, f"../noise_detection_performance_{args.dataset}_{args.model}_{args.label_noises}.png")

    # Plot results
    plot_results(args, f1_scores_ours, f1_scores_baseline, plot_path)


if __name__ == "__main__":
    main()
