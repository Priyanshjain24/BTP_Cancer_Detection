import os, shutil, random, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import *
random.seed(SEED)

class ImageCopier:
    def __init__(self, data_dir, og_dir, cancer_cells, non_cancer_cells, balance_classes):
        self.data_dir = data_dir
        self.og_dir = og_dir
        self.cancer_cells = cancer_cells
        self.non_cancer_cells = non_cancer_cells
        self.balance_classes = balance_classes

        # Create target folders
        self._create_folders()

    def _create_folders(self):
        os.makedirs(os.path.join(self.data_dir, 'cancer'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'non_cancer'), exist_ok=True)

    def collect_images(self, folders):
        images_by_folder = {}
        for folder in folders:
            folder_path = os.path.join(self.og_dir, folder)
            images_by_folder[folder] = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):  # Adjust file types if needed
                        images_by_folder[folder].append(os.path.join(root, file))
        return images_by_folder

    def copy_images(self, images_by_folder, target_folder):
        for folder, images in images_by_folder.items():
            for img in images:
                shutil.copy(img, os.path.join(self.data_dir, target_folder))

    def balance_non_cancer_images(self, non_cancer_images_by_folder, total_cancer_images):
        non_cancer_images_by_folder = dict(sorted(non_cancer_images_by_folder.items(), key=lambda item: len(item[1])))
        selected_non_cancer_images = []
        remaining_folders = list(non_cancer_images_by_folder.keys())
        len_remaining_folder = len(remaining_folders)
        remaining_images_needed = total_cancer_images

        for folder in remaining_folders:
            images_per_folder = remaining_images_needed // len_remaining_folder
            images = non_cancer_images_by_folder[folder]

            if len(images) <= images_per_folder:
                selected_non_cancer_images.extend(images)
                remaining_images_needed -= len(images)
            else:
                selected_images = random.sample(images, images_per_folder)
                selected_non_cancer_images.extend(selected_images)
                remaining_images_needed -= images_per_folder

            len_remaining_folder -= 1

        return selected_non_cancer_images

    def run(self):
        # Collect cancer and non-cancer images
        cancer_images_by_folder = self.collect_images(self.cancer_cells)
        non_cancer_images_by_folder = self.collect_images(self.non_cancer_cells)

        # Copy cancer images
        self.copy_images(cancer_images_by_folder, 'cancer')
        total_cancer_images = sum(len(images) for images in cancer_images_by_folder.values())

        if self.balance_classes:
            # Handle balanced copying for non-cancer images
            selected_non_cancer_images = self.balance_non_cancer_images(non_cancer_images_by_folder, total_cancer_images)
            for img in selected_non_cancer_images:
                shutil.copy(img, os.path.join(self.data_dir, 'non_cancer'))

            print(f"Copied {total_cancer_images} images to 'cancer' folder.")
            print(f"Copied {len(selected_non_cancer_images)} images to 'non_cancer' folder.")
        else:
            # Copy all non-cancer images without balancing
            self.copy_images(non_cancer_images_by_folder, 'non_cancer')
            total_non_cancer_images = sum(len(images) for images in non_cancer_images_by_folder.values())

            print(f"Copied {total_cancer_images} images to 'cancer' folder.")
            print(f"Copied {total_non_cancer_images} images to 'non_cancer' folder.")

if __name__ == "__main__":
    # Example usage
    image_copier = ImageCopier(D1_DATA_DIR, D1_OG_DIR, CANCER_CELLS, NON_CANCER_CELLS, BALANCE)
    image_copier.run()