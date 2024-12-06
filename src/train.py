import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import h5py
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import logging
from datetime import datetime
from utils.model import DDXModel
from utils.metrics import calculate_metrics

def load_sparse_matrix(filename):
    with h5py.File(filename, 'r') as hf:
        data = hf['X'][:]
        indices = hf['indices'][:]
        indptr = hf['indptr'][:]
        shape = hf['shape'][:]
    X_sparse = sparse.csr_matrix((data, indices, indptr), shape=shape)
    return X_sparse

class SparseDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X_sparse = X_sparse
        self.y = y

    def __len__(self):
        return self.X_sparse.shape[0]

    def __getitem__(self, idx):
        # Get the sparse row and convert it to dense
        X_dense = self.X_sparse[idx].toarray().squeeze().astype(np.float32)
        y_item = self.y[idx]
        y_item = torch.tensor(y_item, dtype=torch.long)  # Ensure label is torch.long
        return X_dense, y_item

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train the DDXModel")

    # Paths to data files
    parser.add_argument('--X_train', type=str, default='X_train.h5', help='Path to X_train file')
    parser.add_argument('--y_train', type=str, default='y_train_noisy.npy', help='Path to y_train file')
    parser.add_argument('--X_validate', type=str, default='X_validate.h5', help='Path to X_validate file')
    parser.add_argument('--y_validate', type=str, default='y_validate.npy', help='Path to y_validate file')
    parser.add_argument('--X_test', type=str, default='X_test.h5', help='Path to X_test file')
    parser.add_argument('--y_test', type=str, default='y_test.npy', help='Path to y_test file')

    # Hyperparameters
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[1024, 512], help='List of hidden layer sizes')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')

    # Other parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()

    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'training_{timestamp}.log'

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Remove the default handler to prevent duplicate logs
    logger.removeHandler(logger.handlers[0])

    # Set up the device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Log the hyperparameters
    logger.info('Hyperparameters:')
    logger.info(f'Hidden sizes: {args.hidden_sizes}')
    logger.info(f'Learning rate: {args.learning_rate}')
    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Number of epochs: {args.num_epochs}')
    logger.info(f'Momentum: {args.momentum}')

    # Create the checkpoints directory if it doesn't exist
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load the pathology mapping to get the number of pathologies
    with open('pathology_mapping.pkl', 'rb') as f:
        pathology_mapping = pickle.load(f)
    pathology_to_int = pathology_mapping['pathology_to_int']
    num_pathologies = len(pathology_to_int)
    output_size = num_pathologies  # Number of classes

    # Load the data and ensure labels are int64
    X_train_sparse = load_sparse_matrix(args.X_train)
    y_train = np.load(args.y_train).astype(np.int64)  # Convert to int64

    X_validate_sparse = load_sparse_matrix(args.X_validate)
    y_validate = np.load(args.y_validate).astype(np.int64)  # Convert to int64

    X_test_sparse = load_sparse_matrix(args.X_test)
    y_test = np.load(args.y_test).astype(np.int64)  # Convert to int64

    # Create datasets and data loaders
    batch_size = args.batch_size  # Use batch size from arguments

    train_dataset = SparseDataset(X_train_sparse, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate_dataset = SparseDataset(X_validate_sparse, y_validate)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = SparseDataset(X_test_sparse, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calculate input_size
    input_size = X_train_sparse.shape[1]

    # Instantiate the model with configurable hidden sizes
    hidden_sizes = args.hidden_sizes  # Use hidden sizes from arguments
    model = DDXModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
    model.to(device)  # Move the model to the device

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Set learning rate and momentum for SGD
    learning_rate = args.learning_rate  # Use learning rate from arguments
    momentum = args.momentum  # Use momentum from arguments

    # Initialize the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Define a learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop with validation
    num_epochs = args.num_epochs  # Use number of epochs from arguments
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Step the scheduler
        scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Validation step
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for val_batch_X, val_batch_y in validate_loader:
                # Move data to device
                val_batch_X = val_batch_X.to(device)
                val_batch_y = val_batch_y.to(device)

                val_outputs = model(val_batch_X)
                val_loss = criterion(val_outputs, val_batch_y)
                val_running_loss += val_loss.item() * val_batch_X.size(0)
                _, val_preds = torch.max(val_outputs, 1)
                all_preds.append(val_preds)
                all_labels.append(val_batch_y)

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        val_epoch_loss = val_running_loss / len(validate_loader.dataset)
        val_accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
        val_precision, val_recall, val_f1 = calculate_metrics(all_preds, all_labels, output_size, device)

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"LR: {current_lr:.6f}, "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, "
              f"Val Precision: {val_precision:.4f}, "
              f"Val Recall: {val_recall:.4f}, "
              f"Val F1: {val_f1:.4f}")

        # Save the model checkpoint at each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'learning_rate': current_lr,  # Save the current learning rate
            'loss': epoch_loss,
            'val_loss': val_epoch_loss,
            'val_accuracy': val_accuracy,
        }, checkpoint_path)
        logger.info(f"Model checkpoint saved at {checkpoint_path}")

    # Testing
    model.eval()
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for test_batch_X, test_batch_y in test_loader:
            # Move data to device
            test_batch_X = test_batch_X.to(device)
            test_batch_y = test_batch_y.to(device)

            test_outputs = model(test_batch_X)
            _, test_preds = torch.max(test_outputs, 1)
            all_test_preds.append(test_preds)
            all_test_labels.append(test_batch_y)

    # Concatenate all predictions and labels
    all_test_preds = torch.cat(all_test_preds)
    all_test_labels = torch.cat(all_test_labels)

    # Calculate accuracy
    test_accuracy = (all_test_preds == all_test_labels).sum().item() / len(all_test_labels)

    # Calculate precision, recall, and F1 score
    test_precision, test_recall, test_f1 = calculate_metrics(all_test_preds, all_test_labels, output_size, device)

    logger.info(f"\nTest Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Precision (Macro-average): {test_precision:.4f}")
    logger.info(f"Test Recall (Macro-average): {test_recall:.4f}")
    logger.info(f"Test F1 Score (Macro-average): {test_f1:.4f}")
