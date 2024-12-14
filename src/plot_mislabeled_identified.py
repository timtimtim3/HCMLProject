# =========================
#
#  NEEDS TO BE REFACTORED
#  - split into function
#  - should only be called in files in src
#
# =========================
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend suitable for servers
import matplotlib.pyplot as plt
from datetime import datetime
from utils.parser import add_shared_parser_arguments
from utils.functions import get_output_dir_from_args

def load_self_influences(output_dir, epoch):
    pass

def load_labels(data_dir, label_noise=None):
    if label_noise is None:
        labels = np.load(os.path.join(data_dir, "labels_train.npy"))
    else:
        labels = np.load(os.path.join(data_dir, f"labels_train_noisy{label_noise}.npy"))
    return labels

def aggregate_self_influence_epochs(epochs):
    for epoch in epochs:
        self_influence = load_self_influences(output_dir, epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the influence scores based on a already trained model")
    add_shared_parser_arguments(parser)

    parser.add_argument("--epochs", nargs="+", required=True, type=int,
                        help="Specify which epoch self-influence scores to aggregate and plot")
    parser.add_argument("--label_noises", nargs="+", required=True, type=float,
                        help="Specify which percentages noisy labels to load copmuted self-influences and make plots "
                             "for e.g. 0.1, 0.2, 0.3")

    args = parser.parse_args()

    data_dir = os.path.join("data", args.dataset.upper())
    y_true = load_labels(data_dir, label_noise=None)

    for label_noise in args.label_noises:
        output_dir = get_output_dir_from_args(args, label_noise=label_noise)

        # Load true labels and noisy labels
        y_noisy = load_labels(data_dir, label_noise=label_noise)

        # Load self-influence values
        # self_influence = np.load('self_influence.npy')
        self_influence = aggregate_self_influence_epochs(args.epochs)

        # Identify which samples are noisy (mislabeled)
        noisy_mask = y_true != y_noisy
        noisy_indices = np.where(noisy_mask)[0]
        num_noisy = len(noisy_indices)

        print(f"Total number of samples: {len(y_true)}")
        print(f"Total number of noisy samples: {num_noisy}")

        # Rank samples by self-influence in descending order
        ranked_indices = np.argsort(-self_influence)  # Negative sign for descending order

        # Initialize lists to store results
        fractions_checked = []
        fractions_identified = []

        # Intervals of 10%
        intervals = np.arange(0.1, 1.1, 0.1)  # From 0.1 to 1.0 inclusive
        total_samples = len(y_true)

        for fraction in intervals:
            num_samples_to_check = int(fraction * total_samples)
            top_indices = ranked_indices[:num_samples_to_check]
            # Count how many noisy samples are in top_indices
            num_noisy_in_top = np.intersect1d(top_indices, noisy_indices).size
            # Compute fraction of noisy samples identified
            fraction_identified = num_noisy_in_top / num_noisy
            print(f"Fraction of data checked: {fraction:.0%}, "
                  f"Fraction of mislabeled identified: {fraction_identified:.2%}")
            fractions_checked.append(fraction)
            fractions_identified.append(fraction_identified)

        # Save the results to a PNG file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'self_influence_mislabeled_identification_{timestamp}.png'

        plt.figure(figsize=(8, 6))
        plt.plot(fractions_checked, fractions_identified, marker='o')
        plt.xlabel('Fraction of Data Checked')
        plt.ylabel('Fraction of Mislabeled Identified')
        plt.title('Identification of Mislabeled Samples via Self-Influence')
        plt.xticks(intervals)
        plt.yticks(np.linspace(0, 1, 11))
        plt.grid(True)
        plt.savefig(filename)
        print(f"Plot saved as '{filename}'")
