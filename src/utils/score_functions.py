import os
import json
import numpy as np


def load_self_influences(output_dir, epoch):
    filename = os.path.join(output_dir, f"scores_epoch_{epoch}.json")
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} not found. Make sure the file exists.")

    with open(filename, "r") as f:
        data = json.load(f)
    return data


def aggregate_self_influence_epochs(output_dir, epochs):
    aggregated_scores = {}

    for epoch in epochs:
        data = load_self_influences(output_dir, epoch)
        for entry in data:
            idx = entry["index"]
            score = entry["score"]

            if idx not in aggregated_scores:
                aggregated_scores[idx] = 0.0
            aggregated_scores[idx] += score

    max_index = max(aggregated_scores.keys())
    self_influence = np.zeros(max_index + 1, dtype=float)

    for idx, total_score in aggregated_scores.items():
        self_influence[idx] = total_score

    return self_influence