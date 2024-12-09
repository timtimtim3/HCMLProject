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

    criterion = nn.CrossEntropyLoss()
    
    # Get only model checkpoints
    checkpoint_paths = [p for p in os.listdir(checkpoint_dir) if p.startswith("model_epoch")]

    # Sort by epoch
    sorted_paths = sorted(checkpoint_paths, key=lambda x: (len(x), x))

    # For each checkpoint
    for filename in sorted_paths:

        # Get checkpoint data
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        checkpoint_obj = torch.load(checkpoint_path, map_location=device)        
        checkpoint = ModelCheckpoint(**checkpoint_obj)

        logger.info(f"Loading checkpoint from {checkpoint_path}...")

        # Load into model
        model.load_state_dict(checkpoint.model_state_dict)

        # Scores
        self_influences = []

        for data, labels in tqdm(train_loader, desc=f"Calculating self-influence scores (epoch {checkpoint.epoch})"):

            # Move data to device
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            

            # TODO
            # Calculate self-influences...
            # Add the result to self_influences
            
            # NOTE: every model class has a fully connected layer(s) -> model.fc
            # You can draw inspiration from utils/self_influence.py


            # Temporary
            self_influences.append(loss.item())

        
        output_path = os.path.join(output_dir, f"scores_epoch_{checkpoint.epoch}.json")

        with open(output_path, "w") as f:
            json.dump(self_influences, f, indent=2)

        logger.info(f"Self influence scores saved to {output_path}")
        