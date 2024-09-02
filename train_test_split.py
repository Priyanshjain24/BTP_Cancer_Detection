import os
import shutil
import random
from math import floor

# Set the random seed for reproducibility
random.seed(42)

# Define paths
base_dir = '/home/priyansh/Downloads/datasets/d1_classify/unbalanced'
categories = ['cancer', 'non_cancer']
output_dirs = ['train', 'val', 'test']

# Create train, val, test directories
for output_dir in output_dirs:
    for category in categories:
        os.makedirs(os.path.join(base_dir, output_dir, category), exist_ok=True)

# Function to split data into train, val, and test
def split_data(category, ratio=(0.7, 0.2, 0.1)):
    category_path = os.path.join(base_dir, category)
    files = os.listdir(category_path)
    random.shuffle(files)

    # Calculate the split indices
    train_split = floor(ratio[0] * len(files))
    val_split = floor(ratio[1] * len(files))

    # Split the files
    train_files = files[:train_split]
    val_files = files[train_split:train_split + val_split]
    test_files = files[train_split + val_split:]

    # Move the files to their respective directories
    for file in train_files:
        shutil.move(os.path.join(category_path, file), os.path.join(base_dir, 'train', category, file))

    for file in val_files:
        shutil.move(os.path.join(category_path, file), os.path.join(base_dir, 'val', category, file))

    for file in test_files:
        shutil.move(os.path.join(category_path, file), os.path.join(base_dir, 'test', category, file))

# Apply the split to both categories
for category in categories:
    split_data(category)

# Optional: Remove the original cancer and non_cancer folders if empty
for category in categories:
    category_path = os.path.join(base_dir, category)
    if not os.listdir(category_path):
        os.rmdir(category_path)
