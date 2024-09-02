import os
from PIL import Image

def find_corrupt_images(root_dir):
    corrupt_images = []
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        img.load()  # Fully load the image to catch issues
                except (OSError, Image.DecompressionBombError) as e:
                    print(f"Corrupt image found: {file_path} - Error: {e}")
                    corrupt_images.append(file_path)
    
    return corrupt_images

def delete_files(file_list):
    for file_path in file_list:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    root_dir = '/home/priyansh/Downloads/datasets/d1_classify/unbalanced'
    
    corrupt_images = find_corrupt_images(root_dir)
    
    print(f"\nFound {len(corrupt_images)} corrupt images:")
    for image in corrupt_images:
        print(image)
    
    if corrupt_images:
        delete_confirmation = input("\nDo you want to delete all corrupt images? (yes/no): ").strip().lower()
        if delete_confirmation == 'yes':
            delete_files(corrupt_images)
        else:
            print("No files were deleted.")
    else:
        print("No corrupt images were found.")