import os
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import AVAILABLE_DATASETS
from models import AVAILABLE_MODELS

from utils.dataclasses import ModelCheckpoint
from utils.parser import add_shared_parser_arguments
from utils.functions import get_checkpoint_dir_from_args, get_output_dir_from_args, get_device, get_sample_info, set_seed
from utils.logger import setup_logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the baseline using the current best model")
    add_shared_parser_arguments(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger() 
    device = get_device() 

    # Log parameters
    logger.info(f'Using device: {device}')
    logger.info(f'Args: {vars(args)}')

    checkpoint_dir = get_checkpoint_dir_from_args(args) 
    output_dir = get_output_dir_from_args(args)

    ModelClass = AVAILABLE_MODELS[args.model]
    DatasetClass = AVAILABLE_DATASETS[args.dataset]

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
    checkpoint_obj = torch.load(best_model_path, map_location=device)
    checkpoint = ModelCheckpoint(**checkpoint_obj)

    train_dataset = DatasetClass(split="train", label_noise=args.label_noise, seed=args.seed)
    sample_info = get_sample_info(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model and populate with checkpoint
    model = ModelClass(sample_info, args.hidden_sizes).to(device)
    model.load_state_dict(checkpoint.model_state_dict)

    model.eval()
    results = []

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
            for index, (logit, prob, label) in enumerate(zip(logits, probs, labels)):

                results.append({
                    "index": index,
                    "logits": logit,
                    "probabilities": prob,
                    "label": label
                })


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