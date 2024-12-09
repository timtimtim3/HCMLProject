import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import asdict

from datasets import AVAILABLE_DATASETS
from models import AVAILABLE_MODELS
from models.resnet import get_resnet_transform

from utils.dataclasses import ModelCheckpoint
from utils.parser import add_shared_parser_arguments
from utils.functions import calculate_metrics, get_checkpoint_dir_from_args, get_device, get_sample_info, set_seed
from utils.logger import setup_logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model on a specific dataset")
    add_shared_parser_arguments(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger() 
    device = get_device() 

    # Log parameters
    logger.info(f'Using device: {device}')
    logger.info(f'Args: {vars(args)}')

    checkpoint_dir = get_checkpoint_dir_from_args(args)

    ModelClass = AVAILABLE_MODELS[args.model]
    DatasetClass = AVAILABLE_DATASETS[args.dataset]

    transform = []

    if args.model.startswith("resnet"):
        # The data needs to be transformed if you want to use ResNet, see models/resnet.py
        transform = get_resnet_transform()

        # HACK TO MAKE MNIST WORK WITH A RESNET MODEL, 
        # CAN BE REMOVED AFTER IMPLEMENTATION OF NEW DATASETS

        if args.dataset == "mnist":
            from torchvision import transforms

            # Repeat the graychannel 3 times to get a RGB image of MNIST
            transform.insert(0, transforms.Lambda(lambda x: x.repeat(3,1,1)))


    train_dataset = DatasetClass(split="train", transform=transform, label_noise=args.label_noise)
    val_dataset = DatasetClass(split="val", transform=transform)

    # Initialize model with sample_info (input_size, output_size, etc)
    sample_info = get_sample_info(train_dataset)
    model = ModelClass(sample_info, args.hidden_sizes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize a variable to track the best validation accuracy
    best_val_accuracy = 0.0

    for epoch in range(args.num_epochs):

        # ===== Training =====

        model.train()
        training_loss = 0

        for data, labels in tqdm(train_loader, desc=f"Training epoch {epoch}"):

            # Move data to device
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() / data.size(0)


        # ===== Validation =====

        model.eval()
        validation_loss = 0
        validation_preds = []
        validation_labels = []

        for data, labels in tqdm(val_loader, desc=f"Validating epoch {epoch}"):
            
            data, labels = data.to(device), labels.to(device)

            with torch.no_grad():

                outputs = model(data)
                loss = criterion(outputs, labels)
                validation_loss += loss.item() / data.size(0)

                preds = torch.argmax(outputs, dim=1)

                validation_preds.append(preds)
                validation_labels.append(labels)


        # Concatenate all predictions and labels
        validation_preds = torch.cat(validation_preds)
        validation_labels = torch.cat(validation_labels)

        accuracy, precision, recall, f1 = calculate_metrics(
            validation_preds, 
            validation_labels, 
            val_dataset.NUM_CLASSES,
            device
        )

        logger.info(f"Epoch [{epoch + 1}/{args.num_epochs}], "
              f"Train Loss: {training_loss:.4f}, "
              f"Val Loss: {validation_loss:.4f}, "
              f"Val Acc: {accuracy:.4f}, "
              f"Val Precision: {precision:.4f}, "
              f"Val Recall: {recall:.4f}, "
              f"Val F1: {f1:.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')

        checkpoint = ModelCheckpoint(
            epoch=epoch+1,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            learning_rate=args.lr,
            training_loss=training_loss,
            validation_loss=validation_loss,
            validation_accuracy=accuracy,
            precision_recall_f1=(precision, recall, f1)
        )

        torch.save(asdict(checkpoint), checkpoint_path)

        logger.info(f"Model checkpoint saved at {checkpoint_path}")

        # Update the best model if current accuracy is better
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_model_checkpoint = checkpoint
            logger.info(f"Found new best model with accuracy {accuracy:.4f} at epoch {epoch + 1}")

    # After all epochs are done, save the best model
    if checkpoint is not None:
        best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{checkpoint.epoch}.pth')
        torch.save(asdict(checkpoint), best_model_path)

        logger.info(
            f"Best model from epoch {checkpoint.epoch} saved at {best_model_path} "\
            f"with Val Acc: {best_val_accuracy:.4f}"
        )
