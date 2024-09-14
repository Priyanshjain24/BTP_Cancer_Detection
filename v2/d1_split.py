import os, shutil, random, sys
from math import floor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import *
random.seed(SEED)

class ImageCopierAndSplitter:
    def __init__(self, data_dir, og_dir, cancer_cells, non_cancer_cells, balance_classes, ratio=(0.8, 0.2)):
        self.data_dir = data_dir
        self.og_dir = og_dir
        self.cancer_cells = cancer_cells
        self.non_cancer_cells = non_cancer_cells
        self.balance_classes = balance_classes
        self.ratio = ratio

        # Create train and val target folders with subfolders for each cell type
        self._create_folders()

    def _create_folders(self):
        for split in ['train', 'val']:
            for cell_type in self.cancer_cells:
                os.makedirs(os.path.join(self.data_dir, split, 'cancer', cell_type), exist_ok=True)
            for cell_type in self.non_cancer_cells:
                os.makedirs(os.path.join(self.data_dir, split, 'non_cancer', cell_type), exist_ok=True)

    def collect_images(self, folders):
        images_by_folder = {}
        for folder in folders:
            folder_path = os.path.join(self.og_dir, folder)
            images_by_folder[folder] = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        images_by_folder[folder].append(os.path.join(root, file))
        return images_by_folder

    def split_data(self, images_by_folder):
        split_images = {'train': {}, 'val': {}}
        for folder, images in images_by_folder.items():
            random.shuffle(images)
            split_idx = floor(self.ratio[0] * len(images))
            split_images['train'][folder] = images[:split_idx]
            split_images['val'][folder] = images[split_idx:]
        return split_images

    def copy_images(self, split_images, target_folder):
        for split, folders in split_images.items():
            for folder, images in folders.items():
                subfolder = os.path.join(self.data_dir, split, target_folder, folder)
                for img in images:
                    shutil.copy(img, subfolder)

    def balance_non_cancer_images(self, non_cancer_images_by_folder, total_cancer_images):
        non_cancer_images_by_folder = {
            split: {folder: images for folder, images in folders.items()}
            for split, folders in non_cancer_images_by_folder.items()
        }
        
        selected_non_cancer_images = {'train': {}, 'val': {}}
        
        for split in ['train', 'val']:
            remaining_folders = list(non_cancer_images_by_folder[split].keys())
            len_remaining_folder = len(remaining_folders)
            remaining_images_needed = total_cancer_images[split]

            for folder in remaining_folders:
                images_per_folder = remaining_images_needed // len_remaining_folder
                images = non_cancer_images_by_folder[split][folder]

                if len(images) <= images_per_folder:
                    selected_non_cancer_images[split][folder] = images
                    remaining_images_needed -= len(images)
                else:
                    selected_images = random.sample(images, images_per_folder)
                    selected_non_cancer_images[split][folder] = selected_images
                    remaining_images_needed -= images_per_folder

                len_remaining_folder -= 1

        return selected_non_cancer_images

    def run(self):
        # Collect and split cancer and non-cancer images
        cancer_images_by_folder = self.collect_images(self.cancer_cells)
        non_cancer_images_by_folder = self.collect_images(self.non_cancer_cells)

        cancer_split_images = self.split_data(cancer_images_by_folder)
        non_cancer_split_images = self.split_data(non_cancer_images_by_folder)

        # Copy cancer images into respective folders
        self.copy_images(cancer_split_images, 'cancer')
        total_cancer_images = {
            'train': sum(len(images) for images in cancer_split_images['train'].values()),
            'val': sum(len(images) for images in cancer_split_images['val'].values())
        }

        if self.balance_classes:
            # Handle balanced copying for non-cancer images
            balanced_non_cancer_images = self.balance_non_cancer_images(non_cancer_split_images, total_cancer_images)
            self.copy_images(balanced_non_cancer_images, 'non_cancer')
            print(f"Balanced and copied {total_cancer_images['train']} images to 'train/cancer' and 'train/non_cancer' folders.")
            print(f"Balanced and copied {total_cancer_images['val']} images to 'val/cancer' and 'val/non_cancer' folders.")
        else:
            # Copy all non-cancer images without balancing
            self.copy_images(non_cancer_split_images, 'non_cancer')
            total_non_cancer_images = {
                'train': sum(len(images) for images in non_cancer_split_images['train'].values()),
                'val': sum(len(images) for images in non_cancer_split_images['val'].values())
            }

            print(f"Copied {total_cancer_images['train']} images to 'train/cancer' folder.")
            print(f"Copied {total_non_cancer_images['train']} images to 'train/non_cancer' folder.")
            print(f"Copied {total_cancer_images['val']} images to 'val/cancer' folder.")
            print(f"Copied {total_non_cancer_images['val']} images to 'val/non_cancer' folder.")


if __name__ == "__main__":
    # Example usage
    image_splitter = ImageCopierAndSplitter(D1_DATA_DIR, D1_OG_DIR, CANCER_CELLS, NON_CANCER_CELLS, BALANCE, SPLIT_RATIO)
    image_splitter.run()