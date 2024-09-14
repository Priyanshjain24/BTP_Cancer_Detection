import os, sys
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import *

class ImageHandler:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.corrupt_images = []

    def is_corrupt(self, file_path):
        """Check if the image is corrupt."""
        try:
            with Image.open(file_path) as img:
                img.load()  # Fully load the image to detect corruption
            return False
        except (OSError, Image.DecompressionBombError) as e:
            print(f"Corrupt image found: {file_path} - Error: {e}")
            return True

    def find_corrupt_images(self):
        """Walk through the directory and find corrupt images."""
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    file_path = os.path.join(subdir, file)
                    if self.is_corrupt(file_path):
                        self.corrupt_images.append(file_path)
        return self.corrupt_images

    def delete_files(self):
        """Delete files in the corrupt images list."""
        for file_path in self.corrupt_images:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    # Initialize the image handler
    handler = ImageHandler(D1_OG_DIR)
    
    # Find and delete corrupt images
    corrupt_images = handler.find_corrupt_images()

    if corrupt_images:
        print(f"\nFound {len(corrupt_images)} corrupt images.\n")
        handler.delete_files()
        print("Deleted all corrupt images.")
    else:
        print("No corrupt images were found.")