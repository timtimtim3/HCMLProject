import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import AVAILABLE_DATASETS
from models import AVAILABLE_MODELS
from models.resnet import get_resnet_transform
from utils.functions import calculate_metrics, get_device, set_seed
from utils.logger import setup_logger


models = AVAILABLE_MODELS
datasets = AVAILABLE_DATASETS


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model on a specific dataset")

    parser.add_argument("--dataset", type=str, choices=list(datasets.keys()), required=True)
    parser.add_argument("--model", type=str, choices=list(models.keys()), required=True)
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[50, 50], help='List of hidden layer sizes')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')

    args = parser.parse_args()

    set_seed(42)
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


    checkpoint_dir = f"checkpoints/{args.dataset}_{args.model}_"\
        f"{args.hidden_sizes}_{args.lr}_{args.batch_size}_{args.num_epochs}"

    # Create checkpoint directory
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    ModelClass = models[args.model]
    DatasetClass = datasets[args.dataset]

    transform = []

    if args.model.startswith("resnet"):
        # The data needs to be transformed if you want to use ResNet, see models/resnet.py
        transform = get_resnet_transform()

    train = DatasetClass(split="train", transform=transform, download=True)
    val = DatasetClass(split="val", transform=transform, download=True)

    model = ModelClass(
        input_size=train.data_dim,
        hidden_sizes=args.hidden_sizes,
        output_size=train.label_dim,
        input_channels=train.input_channels
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)

    for epoch in range(args.num_epochs):

        # ===== Training =====

        model.train()
        training_loss = 0

        for data, labels in train_loader:

            # Move data to device
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() / data.size(0)


        # ===== Validation =====

        # NOTE: The validation loop should probably be a separate function
        # since it can also be used to calculate the self_influence

        model.eval()
        validation_loss = 0
        validation_preds = []
        validation_labels = []

        for data, labels in val_loader:
            
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
            val.label_dim,
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

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': args.lr,  # Save the current learning rate
            'loss': training_loss,
            'val_loss': validation_loss,
            'val_accuracy': accuracy,
        }, checkpoint_path)

        logger.info(f"Model checkpoint saved at {checkpoint_path}")
