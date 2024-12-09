import os
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import AVAILABLE_DATASETS
from models import AVAILABLE_MODELS

from utils.parser import add_shared_parser_arguments
from utils.functions import calculate_metrics, get_checkpoint_dir_from_args, get_output_dir_from_args, get_device, set_seed
from utils.logger import setup_logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the influence scores based on a already trained model")
    add_shared_parser_arguments(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger() 
    device = get_device() 

    # Log parameters
    logger.info(f'Using device: {device}')
    logger.info('Using seed 42')
    logger.info('Hyperparameters:')
    logger.info(f'Hidden sizes: {args.hidden_sizes}')
    logger.info(f'Learning rate: {args.lr}')
    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Number of epochs: {args.num_epochs}')

    checkpoint_dir = get_checkpoint_dir_from_args(args) 
    output_dir = get_output_dir_from_args(args)

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ModelClass = AVAILABLE_MODELS[args.model]
    DatasetClass = AVAILABLE_DATASETS[args.dataset]

    # python src/calculate_baseline.py --model cnn --dataset mnist

    # get best model from checkpoint dir (epoch number)
    best_model_path = None
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("best_model_epoch_") and fname.endswith(".pth"):
            best_model_path = os.path.join(checkpoint_dir, fname)
            break

    if best_model_path is None:
        logger.error("No best model checkpoint found in the given checkpoint directory.")
        exit(1)

    logger.info(f"Loading best model checkpoint from {best_model_path}")

    # Load the best model checkpoint
    checkpoint = torch.load(best_model_path, map_location=device)

    train_dataset = DatasetClass(split="train")

    model = ModelClass(
        input_size=train_dataset.data_dim,
        hidden_sizes=args.hidden_sizes,
        output_size=train_dataset.label_dim,
        input_channels=train_dataset.input_channels
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    results = []

    global_index = 0

    # Do inference
    with torch.no_grad():
        for data, labels in tqdm(train_loader, desc="Calculating baseline on training set"):
            data = data.to(device)
            labels = labels.to(device)

            # get logits from model
            logits = model(data)  # [batch_size, num_classes]
            probs = F.softmax(logits, dim=1)  # Convert logits to probabilities

            # Move data back to CPU and convert to lists
            logits = logits.cpu().tolist()
            probs = probs.cpu().tolist()
            labels = labels.cpu().tolist()

            # Store each sample individually
            for logit, prob, label in zip(logits, probs, labels):

                results.append({
                    "index": global_index,
                    "logits": logit,
                    "probabilities": prob,
                    "label": label
                })

                global_index += 1

    # Save the results as a JSON file
    baseline_path = os.path.join(output_dir, "baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Baseline results saved to {baseline_path}")

    # Sort results by the maximum probability (lowest to highest)
    results_sorted = sorted(results, key=lambda x: max(x["probabilities"]))

    # After sorting, assign a sorted_index to each entry
    for i, entry in enumerate(results_sorted):
        entry["sorted_index"] = i

    # Save the sorted results as a separate JSON file
    baseline_sorted_path = os.path.join(output_dir, "baseline_sorted_by_max_prob.json")
    with open(baseline_sorted_path, "w") as f:
        json.dump(results_sorted, f, indent=2)
    logger.info(f"Baseline sorted results saved to {baseline_sorted_path}")