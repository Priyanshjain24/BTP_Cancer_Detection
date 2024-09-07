import os
import shutil
import random
from math import floor
from constants import *
random.seed(SEED)

class DataSplitter:
    def __init__(self, data_dir, classes, splits, ratio=(0.7, 0.2, 0.1)):
        self.data_dir = data_dir
        self.classes = classes
        self.splits = splits
        self.ratio = ratio

    def create_directories(self):
        """Create directories for train, val, test splits."""
        for split in self.splits:
            for category in self.classes:
                os.makedirs(os.path.join(self.data_dir, split, category), exist_ok=True)

    def split_data(self, category):
        """Split the data into train, val, and test sets for a given category."""
        category_path = os.path.join(self.data_dir, category)
        files = os.listdir(category_path)
        random.shuffle(files)

        # Calculate the split indices
        train_split = floor(self.ratio[0] * len(files))
        val_split = floor(self.ratio[1] * len(files))

        # Split the files
        train_files = files[:train_split]
        val_files = files[train_split:train_split + val_split]
        test_files = files[train_split + val_split:]

        # Move the files to their respective directories
        self._move_files(train_files, category, 'train')
        self._move_files(val_files, category, 'val')
        self._move_files(test_files, category, 'test')

    def _move_files(self, files, category, split):
        """Move files to their respective split directories."""
        for file in files:
            shutil.move(os.path.join(self.data_dir, category, file),
                        os.path.join(self.data_dir, split, category, file))

    def remove_empty_folders(self):
        """Remove the original category folders if empty."""
        for category in self.classes:
            category_path = os.path.join(self.data_dir, category)
            if not os.listdir(category_path):
                os.rmdir(category_path)

    def run(self):
        """Run the entire data splitting process."""
        self.create_directories()
        for category in self.classes:
            self.split_data(category)
        self.remove_empty_folders()


# Example usage
if __name__ == "__main__":
    splitter = DataSplitter(data_dir=D1_DATA_DIR, classes=CLASSES, splits=SPLITS)
    splitter.run()