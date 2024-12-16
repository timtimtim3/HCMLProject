import os
import json

def extract_stats(output_path, stats_file):
    """
    Extract stats from a specific stats file.
    :param output_path: Path to the output folder containing JSON files.
    :param stats_file: The specific stats file name.
    :return: Dictionary of stats or None if the file is missing.
    """
    file_path = os.path.join(output_path, stats_file)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

def find_best_epoch(training_stats):
    """
    Identify the best epoch based on validation accuracy.
    :param training_stats: List of epoch statistics.
    :return: Best epoch and its stats.
    """
    best_epoch = max(training_stats, key=lambda x: x["validation_accuracy"])
    return best_epoch

def collect_stats(output_dir):
    """
    Collect stats for ResNet18 and ResNet50 models.
    :param output_dir: Path to the output directory.
    :return: Structured results for ResNet18 and ResNet50.
    """
    resnet18_table = {"0.0": {}, "0.1": {}, "0.2": {}, "0.3": {}, "0.4": {}}
    resnet50_table = {"0.0": {}, "0.1": {}, "0.2": {}, "0.3": {}, "0.4": {}}

    for output_folder in os.listdir(output_dir):
        output_path = os.path.join(output_dir, output_folder)
        if not os.path.isdir(output_path):
            continue

        # Determine model type and noise level
        if "resnet18" in output_folder:
            table = resnet18_table
        elif "resnet50" in output_folder:
            table = resnet50_table
        else:
            continue

        noise_level = output_folder.split("_")[-2]
        if noise_level not in table:
            if "0.0" in output_folder:
                noise_level = "0.0"
            else:
                continue

        # Extract stats for baseline, retraining, and training
        baseline_stats = extract_stats(output_path, "retraining_baseline_stats.json")
        retraining_stats = extract_stats(output_path, f"retraining_{noise_level}_stats.json")
        training_stats = extract_stats(output_path, "training_stats.json")

        if baseline_stats:
            best_baseline = find_best_epoch(baseline_stats)
            table[noise_level]["baseline"] = {
                "Best Epoch": best_baseline["epoch"],
                "Validation Accuracy": best_baseline["validation_accuracy"],
                "F1 Score": best_baseline["f1"],
                "Precision": best_baseline["precision"],
                "Recall": best_baseline["recall"],
                "Training Loss": best_baseline["training_loss"],
                "Validation Loss": best_baseline["validation_loss"]
            }

        if retraining_stats:
            best_retraining = find_best_epoch(retraining_stats)
            table[noise_level]["our_method"] = {
                "Best Epoch": best_retraining["epoch"],
                "Validation Accuracy": best_retraining["validation_accuracy"],
                "F1 Score": best_retraining["f1"],
                "Precision": best_retraining["precision"],
                "Recall": best_retraining["recall"],
                "Training Loss": best_retraining["training_loss"],
                "Validation Loss": best_retraining["validation_loss"]
            }

        if training_stats:
            best_training = find_best_epoch(training_stats)
            table[noise_level]["default_training"] = {
                "Best Epoch": best_training["epoch"],
                "Validation Accuracy": best_training["validation_accuracy"],
                "F1 Score": best_training["f1"],
                "Precision": best_training["precision"],
                "Recall": best_training["recall"],
                "Training Loss": best_training["training_loss"],
                "Validation Loss": best_training["validation_loss"]
            }

    return {"ResNet18": resnet18_table, "ResNet50": resnet50_table}

def save_results_to_json(results, output_file):
    """
    Save the structured results to a JSON file.
    :param results: Dictionary of results for each model.
    :param output_file: Path to the output JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    output_dir = "output"
    output_file = "best_epoch_stats.json"

    results = collect_stats(output_dir)

    # Save results to a JSON file
    save_results_to_json(results, output_file)

    print(f"Results saved to {output_file}")
