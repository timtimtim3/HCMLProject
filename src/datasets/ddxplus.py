import pandas as pd
import json
import numpy as np
import pickle
from scipy import sparse
import h5py

# WE'RE NOT USING THIS

# def load_sparse_matrix(filename):
#     with h5py.File(filename, 'r') as hf:
#         data = hf['X'][:]
#         indices = hf['indices'][:]
#         indptr = hf['indptr'][:]
#         shape = hf['shape'][:]
#     X_sparse = sparse.csr_matrix((data, indices, indptr), shape=shape)
#     return X_sparse

# class SparseDataset(Dataset):
#     def __init__(self, X_sparse, y):
#         self.X_sparse = X_sparse
#         self.y = y

#     def __len__(self):
#         return self.X_sparse.shape[0]

#     def __getitem__(self, idx):
#         # Get the sparse row and convert it to dense
#         X_dense = self.X_sparse[idx].toarray().squeeze().astype(np.float32)
#         y_item = self.y[idx]
#         y_item = torch.tensor(y_item, dtype=torch.long)  # Ensure label is torch.long
#         return X_dense, y_item





# Specify the paths to your JSON files
evidence_file_path = "ddxplus/release_evidences.json"
pathologies_file_path = "ddxplus/release_conditions.json"

# Load the JSON files
with open(evidence_file_path, "r", encoding="utf-8") as file:
    evidence_data = json.load(file)

with open(pathologies_file_path, "r", encoding="utf-8") as file:
    pathology_data = json.load(file)

# Step 1: Create a mapping from evidence strings to integers
evidence_keys = []
evidence_types = {}  # To store the data type of each evidence

binary_count = 0
multi_count = 0
multi_total = 0

for key, entry in evidence_data.items():
    data_type = entry.get('data_type')
    evidence_types[key] = data_type
    possible_values = entry.get('possible-values', [])
    n_values = len(possible_values)
    n_values = n_values if n_values > 0 else 1  # For binary evidences with no possible-values
    if data_type == 'B':
        # Binary evidence, use the key itself
        evidence_keys.append(key)
        binary_count += 1
    elif data_type in ('M', 'C'):
        # Multi or categorical evidence, create keys for each possible value
        for val in possible_values:
            # Create the evidence string
            evidence_str = f"{key}_@_{val}"
            evidence_keys.append(evidence_str)
        multi_total += n_values
        multi_count += 1
    else:
        # Handle any unexpected data types if necessary
        print(f"Unknown data type {data_type} for key {key}")

# Create a mapping from evidence strings to integers
evidence_to_int = {evidence_key: idx for idx, evidence_key in enumerate(evidence_keys)}
int_to_evidence = {idx: evidence_key for evidence_key, idx in evidence_to_int.items()}

# Save the evidence mapping
with open('evidence_mapping.pkl', 'wb') as f:
    pickle.dump({'evidence_to_int': evidence_to_int, 'int_to_evidence': int_to_evidence}, f)

# Step 2: Create a mapping from pathology names to integers
pathology_keys = list(pathology_data.keys())
num_pathologies = len(pathology_keys)
pathology_to_int = {pathology: idx for idx, pathology in enumerate(pathology_keys)}
int_to_pathology = {idx: pathology for pathology, idx in pathology_to_int.items()}

# Save the pathology mapping
with open('pathology_mapping.pkl', 'wb') as f:
    pickle.dump({'pathology_to_int': pathology_to_int, 'int_to_pathology': int_to_pathology}, f)

# Print the counts
print(f"Binary evidences: {binary_count}")
print(f"Multi/Categorical evidences: {multi_count}")
print(f"Total multi values: {multi_total}")
print(f"Total number of pathologies: {num_pathologies}")

# Prepare to process each data split
splits = {'train': 'train.csv', 'test': 'test.csv', 'validate': 'validate.csv'}
max_ages = {}  # To store the max age for each split


# Function to parse the 'EVIDENCES' column
def parse_evidences(evidence_str):
    if isinstance(evidence_str, str):
        # Assuming the evidence_str is a string representation of a list
        return eval(evidence_str)
    elif isinstance(evidence_str, list):
        return evidence_str
    else:
        return []


# Process each split
for split_name, filename in splits.items():
    print(f"\nProcessing {split_name} split...")
    # Load the dataset
    df = pd.read_csv("hf://datasets/aai530-group6/ddxplus/" + filename)

    # Convert 'AGE' to a normalized float between 0 and 1
    max_age = df['AGE'].max()
    max_ages[split_name] = max_age  # Store the max age for this split
    df['AGE_NORM'] = df['AGE'] / max_age  # Normalize within the split

    # Convert 'SEX' to a binary variable (e.g., 0 for 'M', 1 for 'F')
    df['SEX_BIN'] = df['SEX'].map({'M': 0, 'F': 1}).astype(np.int8)

    # Map 'PATHOLOGY' to integers
    df['PATHOLOGY_INT'] = df['PATHOLOGY'].map(pathology_to_int)

    # Handle any unmapped pathologies (optional)
    unmapped_pathologies = df[df['PATHOLOGY_INT'].isnull()]['PATHOLOGY'].unique()
    if len(unmapped_pathologies) > 0:
        print(f"Unmapped pathologies found in {split_name}: {unmapped_pathologies}")

    # Process 'EVIDENCES' column
    # Initialize lists to construct sparse matrix
    rows = []
    cols = []
    data = []

    df['EVIDENCES_LIST'] = df['EVIDENCES'].apply(parse_evidences)

    for idx, evidence_list in enumerate(df['EVIDENCES_LIST']):
        if not isinstance(evidence_list, list):
            print(f"Invalid evidence list at index {idx} in {split_name}: {evidence_list}")
            continue

        for evidence in evidence_list:
            # Map the evidence to an integer index
            evidence_idx = evidence_to_int.get(evidence)
            if evidence_idx is not None:
                rows.append(idx)
                cols.append(evidence_idx)
                data.append(1)
            else:
                # Handle cases where the evidence is not found in the mapping
                print(f"Unknown evidence '{evidence}' at index {idx} in {split_name}")

    # Create sparse matrix for evidences
    num_samples = len(df)
    num_evidences = len(evidence_keys)
    evidence_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(num_samples, num_evidences), dtype=np.int8)

    # Convert 'AGE_NORM' to float32
    age_norm = df['AGE_NORM'].astype(np.float32).values.reshape(-1, 1)
    sex_bin = df['SEX_BIN'].values.reshape(-1, 1)  # Already int8

    # Stack age and sex with sparse evidences
    # Since age and sex are dense, we need to convert them to sparse format
    age_sparse = sparse.csr_matrix(age_norm)
    sex_sparse = sparse.csr_matrix(sex_bin)

    # Concatenate all features horizontally
    X_sparse = sparse.hstack([age_sparse, sex_sparse, evidence_sparse], format='csr')

    # Get labels
    y = df['PATHOLOGY_INT'].values.astype(np.int16)  # Use smaller integer type if possible

    # Save 'X' and 'y' with filenames indicating the split
    # Using HDF5 format for efficient storage
    with h5py.File(f'X_{split_name}.h5', 'w') as hf:
        hf.create_dataset('X', data=X_sparse.data, compression='gzip')
        hf.create_dataset('indices', data=X_sparse.indices, compression='gzip')
        hf.create_dataset('indptr', data=X_sparse.indptr, compression='gzip')
        hf.create_dataset('shape', data=X_sparse.shape)
    np.save(f'y_{split_name}.npy', y)

    print(f"Saved X_{split_name}.h5 and y_{split_name}.npy")

    # Optionally, print shapes to verify
    print(f"X_{split_name}.shape: {X_sparse.shape}")
    print(f"y_{split_name}.shape: {y.shape}")

# Calculate input_size and output_size
input_size = X_sparse.shape[1]  # Includes AGE_NORM and SEX_BIN
output_size = num_pathologies  # Number of pathologies

# Print input_size and output_size
print(f"\nFinal Input size: {input_size}")
print(f"Output size: {output_size}")
