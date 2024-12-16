import argparse
import os
import json
import numpy as np
import matplotlib

from utils.score_functions import aggregate_self_influence_epochs
matplotlib.use('Agg')  # Use a non-interactive backend suitable for servers
import matplotlib.pyplot as plt
from utils.parser import add_shared_parser_arguments
from utils.functions import get_output_dir_from_args

# Normalized colors for professional styling
red_resnet18 = (0.7686, 0.1176, 0.2275, 0.3)
blue_resnet18 = (0.3922, 0.5843, 0.9294, 0.3)
red_resnet50 = (0.7686, 0.1176, 0.2275, 0.9)
blue_resnet50 = (0.3922, 0.5843, 0.9294, 0.9)
purple = (0.5804, 0.0, 0.8275, 0.9)
pink = (1.0, 0.7529, 0.7961, 0.9)


def load_labels(data_dir, label_noise=None):
    if label_noise is None:
        labels = np.load(os.path.join(data_dir, "labels_train.npy"))
    else:
        labels = np.load(os.path.join(data_dir, f"labels_train_noisy{label_noise}.npy"))
    return labels

def load_baseline_sorted_indices(output_dir):
    """
    Loads baseline_sorted_by_max_prob.json from the given output_dir and
    returns an array of indices sorted from low to high confidence.
    """
    baseline_file = os.path.join(output_dir, "baseline_sorted_by_max_prob.json")
    if not os.path.isfile(baseline_file):
        raise FileNotFoundError(f"{baseline_file} not found. Ensure that baseline file is in {output_dir}")

    with open(baseline_file, 'r') as f:
        data = json.load(f)
    baseline_indices = np.array([entry["index"] for entry in data])
    return baseline_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the influence scores based on an already trained model")
    add_shared_parser_arguments(parser)

    parser.add_argument("--epochs", nargs="+", required=True, type=int,
                        help="Specify which epoch self-influence scores to aggregate and plot")
    parser.add_argument("--label_noises", nargs="+", required=True, type=float,
                        help="Specify which percentages noisy labels to load computed self-influences and make plots "
                             "for e.g. 0.1, 0.2, 0.3")

    args = parser.parse_args()

    if args.model == 'resnet18':
        model_name = "ResNet18"
    elif args.model == 'resnet50':
        model_name = "ResNet50"

    data_dir = os.path.join("data", args.dataset.upper())
    y_true = load_labels(data_dir, label_noise=None)

    # We'll store data for combined plotting
    combined_label_noises_data = []

    for i, label_noise in enumerate(args.label_noises):
        output_dir = get_output_dir_from_args(args, label_noise=label_noise)

        # Load true labels and noisy labels
        y_noisy = load_labels(data_dir, label_noise=label_noise)

        # Aggregate self-influence values
        self_influence = aggregate_self_influence_epochs(output_dir, [args.epochs[i]])

        # Identify which samples are noisy (mislabeled)
        noisy_mask = y_true != y_noisy
        noisy_indices = np.where(noisy_mask)[0]
        num_noisy = len(noisy_indices)

        print(f"Label noise: {label_noise}")
        print(f"Total number of samples: {len(y_true)}")
        print(f"Total number of noisy samples: {num_noisy}")

        # Rank samples by self-influence in descending order
        ranked_indices = np.argsort(-self_influence)

        # Load the baseline ordering for this specific label_noise
        baseline_indices = load_baseline_sorted_indices(output_dir)

        # Intervals of 10%
        intervals = np.arange(0.1, 1.1, 0.1)
        total_samples = len(y_true)

        # Calculate fractions for self-influence ranking
        fractions_checked = []
        fractions_identified = []
        for fraction in intervals:
            num_samples_to_check = int(fraction * total_samples)
            top_indices = ranked_indices[:num_samples_to_check]
            # Count how many noisy samples are in top_indices
            num_noisy_in_top = np.intersect1d(top_indices, noisy_indices).size
            fraction_id = num_noisy_in_top / num_noisy if num_noisy > 0 else 0.0
            print(f"Fraction of data checked: {fraction:.0%}, "
                  f"Fraction of mislabeled identified (self-influence): {fraction_id:.2%}")
            fractions_checked.append(fraction)
            fractions_identified.append(fraction_id)

        # Calculate fractions for baseline ranking
        baseline_fractions_identified = []
        for fraction in intervals:
            num_samples_to_check = int(fraction * total_samples)
            top_indices = baseline_indices[:num_samples_to_check]
            num_noisy_in_top = np.intersect1d(top_indices, noisy_indices).size
            fraction_id = num_noisy_in_top / num_noisy if num_noisy > 0 else 0.0
            print(f"Fraction of data checked: {fraction:.0%}, "
                  f"Fraction of mislabeled identified (baseline): {fraction_id:.2%}")
            baseline_fractions_identified.append(fraction_id)

        # Save the mislabeled identification plot
        save_path = os.path.join(output_dir, f"../self_influence_mislabeled_identification_{args.dataset}_{args.model}_noise_{label_noise}.png")

        plt.figure(figsize=(8, 6))
        # Plot self-influence line
        plt.plot(fractions_checked, fractions_identified, color=red_resnet50, linewidth=2, label=f"Self-Influence ({model_name})")
        # Plot baseline line
        plt.plot(fractions_checked, baseline_fractions_identified, color=blue_resnet50, linewidth=2, label=f"Baseline ({model_name})")

        plt.xlabel('Fraction of Data Checked', fontsize=14)
        plt.ylabel('Fraction of Mislabeled Identified', fontsize=14)
        plt.title(f'Mislabeled Identification Performance on {args.dataset.upper()} using {model_name}', fontsize=16)
        plt.xticks(intervals, fontsize=12)
        plt.yticks(np.linspace(0, 1, 11), fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, loc='lower right')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")

        # Now create the self-influence score curve
        sorted_self_influence = self_influence[ranked_indices]

        # Sample ~1000 points
        num_points = 1000
        if len(sorted_self_influence) < num_points:
            sampled_scores = sorted_self_influence
            x_vals = np.arange(len(sampled_scores))
        else:
            sample_indices = np.linspace(0, len(sorted_self_influence) - 1, num=num_points, dtype=int)
            sampled_scores = sorted_self_influence[sample_indices]
            x_vals = sample_indices

        # Convert indices to percentage of total samples
        x_pct = (x_vals / total_samples) * 100

        # Save an individual plot for each label_noise
        curve_save_path = os.path.join(output_dir, f"../self_influence_curve_{args.dataset}_{args.model}_noise_{label_noise}.png")

        plt.figure(figsize=(8, 6))
        plt.plot(x_pct, sampled_scores, color=red_resnet50, linewidth=2, label=f'Noise: {label_noise}')
        plt.xlabel('Percentage of Data', fontsize=14)
        plt.ylabel('Self-Influence Score', fontsize=14)
        plt.title(f'Self-Influence Score Curve on {args.dataset.upper()} using {model_name}', fontsize=16)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(curve_save_path, bbox_inches='tight')
        print(f"Self-influence curve plot saved as '{curve_save_path}'")

        # Store for combined plot
        combined_label_noises_data.append((label_noise, x_pct, sampled_scores))

    # For the combined plot:
    if len(args.label_noises) > 1:
        combined_save_path = os.path.join(output_dir, f"../self_influence_curve_combined_{args.dataset}_{args.model}.png")

        colors = [blue_resnet50, red_resnet50, purple, pink]

        plt.figure(figsize=(8, 6))
        for j, (ln, x_pct_j, sampled_scores_j) in enumerate(combined_label_noises_data):
            color = colors[j]
            plt.plot(x_pct_j, sampled_scores_j, color=color, linewidth=2, label=f'Noise: {ln}')
        plt.xlabel('Percentage of Data', fontsize=14)
        plt.ylabel('Self-Influence Score', fontsize=14)
        plt.title(f'Self-Influence Score Curves on {args.dataset.upper()} using {model_name}', fontsize=16)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(combined_save_path, bbox_inches='tight')
        print(f"Combined self-influence curve plot saved as '{combined_save_path}'")
