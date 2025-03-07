import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def compute_mean_variance(root_dir, output_file="stats.txt"):
    transform = transforms.ToTensor()
    image_paths = []
    
    # Collect all TIFF image paths
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".tif"):
                image_paths.append(os.path.join(subdir, file))
    
    num_pixels = 0
    mean = torch.zeros(3)
    M2 = torch.zeros(3)  # Variance calculation using Welford's method
    
    # Process images with a progress bar
    for img_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        image = Image.open(img_path).convert("RGB")  # Ensure it's RGB
        tensor_img = transform(image)
        
        pixels = tensor_img.view(3, -1)
        num_new_pixels = pixels.shape[1]
        num_pixels += num_new_pixels
        
        new_mean = pixels.mean(dim=1)
        delta = new_mean - mean
        mean += delta * (num_new_pixels / num_pixels)
        M2 += ((pixels - mean[:, None]) ** 2).sum(dim=1)
    
    variance = M2 / num_pixels
    
    mean_rounded = [round(x.item(), 3) for x in mean]
    variance_rounded = [round(x.item(), 3) for x in variance]
    
    stats = f"Mean: {mean_rounded}\nVariance: {variance_rounded}"
    print(stats)
    
    with open(output_file, "w") as f:
        f.write(stats + "\n")

if __name__ == "__main__":
    root_directory = "/home/Drivehd2tb/garima/datasets/og_data/PKG - AML-Cytomorphology_MLL_Helmholtz_v1"  
    compute_mean_variance(root_directory, output_file='d2_stats.txt')