import json
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import AVAILABLE_DATASETS
from models import AVAILABLE_MODELS
from utils.dataclasses import ModelCheckpoint
from utils.functions import get_checkpoint_dir_from_args, get_device, get_output_dir_from_args, get_sample_info, set_seed
from utils.logger import setup_logger
from utils.parser import add_shared_parser_arguments


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the influence scores based on a already trained model")
    add_shared_parser_arguments(parser)

    parser.add_argument("--epochs", nargs="+", required=True, type=int, help="Specify for which epochs to calculate the influence scores")

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

    train_dataset = DatasetClass(split="train", label_noise=args.label_noise)
    sample_info = get_sample_info(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    model = ModelClass(sample_info, args.hidden_sizes).to(device)

    # Set reduction to None to get the loss for each sample
    criterion = nn.CrossEntropyLoss(reduction="none")
    
    # Get only model checkpoints
    checkpoint_paths = [p for p in os.listdir(checkpoint_dir) if p.startswith("model_epoch")]

    # Sort by epoch
    sorted_paths = sorted(checkpoint_paths, key=lambda x: (len(x), x))

    selected_epochs = [f"model_epoch_{e}.pth" for e in args.epochs]

    # For each checkpoint
    for filename in selected_epochs:

        # Get checkpoint data
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        checkpoint_obj = torch.load(checkpoint_path, map_location=device)        
        checkpoint = ModelCheckpoint(**checkpoint_obj)

        logger.info(f"Loading checkpoint from {checkpoint_path}...")

        # Load into model
        model.load_state_dict(checkpoint.model_state_dict)

        # Scores
        self_influences = []
        model_outputs = []
        model_loss = []
        all_labels = []

        for data, labels in tqdm(train_loader, desc=f"Calculating self-influence scores (epoch {checkpoint.epoch})"):

            # Move data to device
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            outputs_list = outputs.detach().cpu().numpy().tolist()
            loss_list = loss.detach().cpu().numpy().tolist()
            labels_list = labels.detach().cpu().numpy().tolist()

            model_outputs.extend(outputs_list)
            model_loss.extend(loss_list)
            all_labels.extend(labels_list)


            for sample_loss in loss:

                model.zero_grad()

                # Set retain_graph to True to access gradients after they have been freed
                sample_loss.backward(retain_graph=True)

                # Get gradients from the last fully connected layer
                grads = [g.grad for g in model.fc[-1].parameters()]
                flat_grads = torch.cat([g.view(-1) for g in grads])

                # Calculate score and add to list
                score = checkpoint.learning_rate * torch.dot(flat_grads, flat_grads)
                self_influences.append(score.cpu().item())
            

        results = []

        for index, (score, logits, loss, label) in enumerate(zip(self_influences, model_outputs, model_loss, all_labels)):
            results.append(
                {
                    "index": index,
                    "score": score,
                    "logits": logits,
                    "loss": loss,
                    "label": label
                }   
            )

        output_path = os.path.join(output_dir, f"scores_epoch_{checkpoint.epoch}.json")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Self influence scores saved to {output_path}")

        # Sort results based on score (low to high)
        results_sorted = sorted(results, key=lambda x: x["score"])

        # Add sorted into to results
        for i, entry in enumerate(results_sorted):
            entry["sorted_index"] = i

        sorted_output_path = os.path.join(output_dir, f"scores_sorted_epoch_{checkpoint.epoch}.json")
        
        with open(sorted_output_path, "w") as f:
            json.dump(results_sorted, f, indent=2)

        logger.info(f"Self influence scores saved to {output_path}")
       