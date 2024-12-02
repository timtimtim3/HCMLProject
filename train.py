import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import h5py
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
from model import DDXModel
from metrics import calculate_metrics


# Set up the device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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
y_train = np.load('y_train.npy').astype(np.int64)  # Convert to int64

X_validate_sparse = load_sparse_matrix('X_validate.h5')
y_validate = np.load('y_validate.npy').astype(np.int64)  # Convert to int64

X_test_sparse = load_sparse_matrix('X_test.h5')
y_test = np.load('y_test.npy').astype(np.int64)  # Convert to int64


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


# Create datasets and data loaders
batch_size = 64  # Adjust as needed to avoid memory issues

train_dataset = SparseDataset(X_train_sparse, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validate_dataset = SparseDataset(X_validate_sparse, y_validate)
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

test_dataset = SparseDataset(X_test_sparse, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Calculate input_size
input_size = X_train_sparse.shape[1]

# Instantiate the model
hidden_sizes = [2048, 1024, 512]  # Adjust as needed
model = DDXModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
model.to(device)  # Move the model to the device

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop with validation
num_epochs = 10
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

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, "
          f"Val Acc: {val_accuracy:.4f}, "
          f"Val Precision: {val_precision:.4f}, "
          f"Val Recall: {val_recall:.4f}, "
          f"Val F1: {val_f1:.4f}")

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

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Precision (Macro-average): {test_precision:.4f}")
print(f"Test Recall (Macro-average): {test_recall:.4f}")
print(f"Test F1 Score (Macro-average): {test_f1:.4f}")
