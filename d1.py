import os
import shutil
import random

# Directories
source_dir = "/home/priyansh/Downloads/datasets/PKG - Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer_v1/Bone-Marrow-Cytomorphology/jpgs/BM_cytomorphology_data"
target_dir = "/home/priyansh/Downloads/datasets/d1_classify/unbalanced"
cancer_folders = ["BLA", "FGC"]
non_cancer_folders = ["ART", "BAS", "EOS", "MMZ", "MON", "MYB", "NGB", "NGS", "OTH", "LYT", "NIF", "PLM"]

# Balance classes or not and set the seed value for reproducibility
balance_classes = False
seed_value = 42
random.seed(seed_value)

# Create target folders
os.makedirs(os.path.join(target_dir, 'cancer'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'non_cancer'), exist_ok=True)

# Function to collect all image paths from the given folders
def collect_images(folders):
    images_by_folder = {}
    for folder in folders:
        folder_path = os.path.join(source_dir, folder)
        images_by_folder[folder] = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):  # Adjust the file types as needed
                    images_by_folder[folder].append(os.path.join(root, file))
    return images_by_folder

# Collect images
cancer_images_by_folder = collect_images(cancer_folders)
non_cancer_images_by_folder = collect_images(non_cancer_folders)

# Copy all cancer images
for folder, images in cancer_images_by_folder.items():
    for img in images:
        shutil.copy(img, os.path.join(target_dir, 'cancer'))

# Calculate the total number of cancer images
total_cancer_images = sum(len(images) for images in cancer_images_by_folder.values())

# Handle non-cancer images based on the balance_classes flag
if balance_classes:
    # Sort non-cancer folders by the number of images in ascending order
    non_cancer_images_by_folder = dict(sorted(non_cancer_images_by_folder.items(), key=lambda item: len(item[1])))

    # Initialize selected non-cancer images list
    selected_non_cancer_images = []

    # Distribute images from non-cancer folders, adjusting for folders with fewer images
    list_remaining_folders = list(non_cancer_images_by_folder.keys())
    len_remaining_folder = len(list_remaining_folders)
    remaining_images_needed = total_cancer_images

    for folder in list_remaining_folders:
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

    # Copy selected non-cancer images
    for img in selected_non_cancer_images:
        shutil.copy(img, os.path.join(target_dir, 'non_cancer'))

    print(f"Copied {total_cancer_images} images to 'cancer' folder.")
    print(f"Copied {len(selected_non_cancer_images)} images to 'non_cancer' folder.")
else:
    # Copy all non-cancer images without balancing
    for folder, images in non_cancer_images_by_folder.items():
        for img in images:
            shutil.copy(img, os.path.join(target_dir, 'non_cancer'))

    total_non_cancer_images = sum(len(images) for images in non_cancer_images_by_folder.values())
    print(f"Copied {total_cancer_images} images to 'cancer' folder.")
    print(f"Copied {total_non_cancer_images} images to 'non_cancer' folder.")