
# =========================
# 
#  NEEDS TO BE REFACTORED
#  - split into function
#  - should only be called in the get_influences.py
# 
# =========================


import torch
import torch.nn as nn
import numpy as np
import pickle
import h5py
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
import os
from model import DDXModel  # Ensure this matches your model definition
from tqdm import tqdm  # For progress bars
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Compute Self-Influence using TrackInCP")
parser.add_argument('--num_checkpoints', type=int, default=1, help='Number of checkpoints to use (default: 10)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory containing checkpoints')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loaders (default: 32)')
parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[1024, 512], help='List of hidden layer sizes')
args = parser.parse_args()

# Set up the device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Directory containing checkpoints
checkpoint_dir = args.checkpoint_dir

# Load the pathology mapping to get the number of pathologies
with open('pathology_mapping.pkl', 'rb') as f:
    pathology_mapping = pickle.load(f)
pathology_to_int = pathology_mapping['pathology_to_int']
num_pathologies = len(pathology_to_int)
output_size = num_pathologies  # Number of classes

# Function to load sparse matrix from HDF5 file
def load_sparse_matrix(filename):
    with h5py.File(filename, 'r') as hf:
        data = hf['X'][:]
        indices = hf['indices'][:]
        indptr = hf['indptr'][:]
        shape = hf['shape'][:]
    X_sparse = sparse.csr_matrix((data, indices, indptr), shape=shape)
    return X_sparse

# Load the data and ensure labels are int64
X_train_sparse = load_sparse_matrix('X_train.h5')
y_train = np.load('y_train_noisy.npy').astype(np.int64)  # Convert to int64

# Define the SparseDataset class
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
        return X_dense, y_item

# Create dataset and data loader
batch_size = args.batch_size  # Use batch size from arguments
train_dataset = SparseDataset(X_train_sparse, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Calculate input_size
input_size = X_train_sparse.shape[1]

# Get list of checkpoint files
checkpoint_files = sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
                           if f.startswith('model_epoch_') and f.endswith('.pth')])

# Use the first X checkpoints
num_checkpoints_to_use = args.num_checkpoints
checkpoint_files = checkpoint_files[:num_checkpoints_to_use]

if len(checkpoint_files) == 0:
    print("No checkpoints found. Exiting.")
    exit()

print(f"Using the first {len(checkpoint_files)} checkpoints from '{checkpoint_dir}'.")

# Initialize self-influence array
num_samples = len(train_dataset)
self_influence = np.zeros(num_samples)

# Function to compute per-sample gradients
def compute_per_sample_gradients(model, inputs, targets):
    # Ensure model is in evaluation mode
    model.eval()
    # Set requires_grad for parameters
    for param in model.parameters():
        param.requires_grad = True
    # Zero gradients
    model.zero_grad()
    # Move data to device
    inputs = inputs.to(device)
    targets = targets.to(device)
    # Compute outputs
    outputs = model(inputs)
    # Compute per-sample losses
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = criterion(outputs, targets)
    # Compute gradients
    grads = []
    for i in range(len(losses)):
        # Zero gradients
        model.zero_grad()
        # Compute gradient for sample i
        losses[i].backward(retain_graph=True)
        grad = []
        for param in model.parameters():
            if param.grad is not None:
                grad.append(param.grad.detach().clone().cpu())
            else:
                # In case some parameters do not have gradients
                grad.append(torch.zeros_like(param).cpu())
        grads.append(grad)
    # Convert grads to list of tensors
    return grads

# Main loop over checkpoints
for checkpoint_path in tqdm(checkpoint_files, desc='Processing checkpoints'):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    epoch = checkpoint['epoch']
    learning_rate = checkpoint.get('learning_rate', None)
    if learning_rate is None:
        # Get learning rate from optimizer state_dict
        learning_rate = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    # Initialize model with configurable hidden sizes
    model = DDXModel(input_size=input_size, hidden_sizes=args.hidden_sizes, output_size=output_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # Process training data in batches
    sample_idx = 0  # To keep track of sample indices
    for inputs, targets in tqdm(train_loader, desc=f'Checkpoint {epoch}', leave=False):
        batch_size_actual = inputs.size(0)
        # Compute per-sample gradients
        per_sample_grads = compute_per_sample_gradients(model, inputs, targets)
        # Compute influence for each sample in the batch
        for i in range(batch_size_actual):
            grad = per_sample_grads[i]
            # Flatten and concatenate all gradients into a single vector
            grad_vector = torch.cat([g.view(-1) for g in grad]).cpu()
            # Compute dot product (since z = z')
            influence = learning_rate * torch.dot(grad_vector, grad_vector).item()
            # Accumulate influence
            self_influence[sample_idx] += influence
            sample_idx += 1
    # Cleanup
    del model
    torch.cuda.empty_cache()

# Save self-influence values
np.save('self_influence.npy', self_influence)
print("Saved self-influence values to 'self_influence.npy'")
