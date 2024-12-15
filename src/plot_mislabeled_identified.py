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

    data_dir = os.path.join("data", args.dataset.upper())
    y_true = load_labels(data_dir, label_noise=None)

    # We'll store data for combined plotting
    combined_label_noises_data = []
    colors = ['blue', 'red', 'gray', 'green', 'orange', 'purple']
    # Just ensure we have enough colors for the number of label_noises
    # If not, they will just repeat after these colors.

    for i, label_noise in enumerate(args.label_noises):
        output_dir = get_output_dir_from_args(args, label_noise=label_noise)

        # Load true labels and noisy labels
        y_noisy = load_labels(data_dir, label_noise=label_noise)

        # Aggregate self-influence values
        self_influence = aggregate_self_influence_epochs(output_dir, args.epochs)

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
        epochs_str = "_".join(map(str, args.epochs))
        filename = f'self_influence_mislabeled_identification_noise{label_noise}_epochs{epochs_str}.png'
        save_path = os.path.join(output_dir, filename)

        plt.figure(figsize=(8, 6))
        # Plot self-influence line in blue
        plt.plot(fractions_checked, fractions_identified, label='Self-Influence')
        # Plot baseline line in red
        plt.plot(fractions_checked, baseline_fractions_identified, color='red', label='Baseline Max Prob')

        plt.xlabel('Fraction of Data Checked')
        plt.ylabel('Fraction of Mislabeled Identified')
        plt.title('Identification of Mislabeled Samples')
        plt.xticks(intervals)
        plt.yticks(np.linspace(0, 1, 11))
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        print(f"Plot saved as '{save_path}'")

        # Now create the self-influence score curve
        # Sort self-influence in descending order
        sorted_self_influence = self_influence[ranked_indices]

        total_samples = len(y_true)  # Make sure this is defined at the top level

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
        curve_filename = f'self_influence_curve_noise{label_noise}_epochs{epochs_str}.png'
        curve_save_path = os.path.join(output_dir, curve_filename)

        plt.figure(figsize=(8, 6))
        plt.plot(x_pct, sampled_scores, color=colors[i % len(colors)], label=f'Noise: {label_noise}')
        plt.xlabel('Percentage of Data')
        plt.ylabel('Self-Influence Score')
        plt.title('Self-Influence Score Curve')
        plt.grid(True)
        plt.legend()
        plt.savefig(curve_save_path)
        print(f"Self-influence curve plot saved as '{curve_save_path}'")

        # Store for combined plot
        combined_label_noises_data.append((label_noise, x_pct, sampled_scores))

    # For the combined plot:
    if len(args.label_noises) > 1:
        combined_filename = f'self_influence_curve_noise{"_".join(map(str, args.label_noises))}_epochs{epochs_str}.png'
        first_output_dir = get_output_dir_from_args(args, label_noise=args.label_noises[0])
        combined_save_path = os.path.join(first_output_dir, combined_filename)

        plt.figure(figsize=(8, 6))
        for j, (ln, x_pct_j, sampled_scores_j) in enumerate(combined_label_noises_data):
            plt.plot(x_pct_j, sampled_scores_j, color=colors[j % len(colors)], label=f'Noise: {ln}')
        plt.xlabel('Percentage of Data')
        plt.ylabel('Self-Influence Score')
        plt.title('Self-Influence Score Curves (Combined)')
        plt.grid(True)
        plt.legend()
        plt.savefig(combined_save_path)
        print(f"Combined self-influence curve plot saved as '{combined_save_path}'")
