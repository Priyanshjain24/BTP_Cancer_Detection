import os
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import *

def process_directories(input_dir, output_dir, mutations):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    else:
        os.makedirs(output_dir)
        
    for mutation in mutations:
        mutation_input_folder = os.path.join(input_dir, mutation)
        mutation_output_folder = os.path.join(output_dir, mutation)
        os.makedirs(mutation_output_folder, exist_ok=True)

        for patient in os.listdir(mutation_input_folder):
            patient_input_folder = os.path.join(mutation_input_folder, patient)
            cancer_input_folder = os.path.join(patient_input_folder, 'cancer')

            for image in os.listdir(cancer_input_folder):
                src_image_path = os.path.join(cancer_input_folder, image)
                dst_image_path = os.path.join(mutation_output_folder, f"{patient}_{image.split('_')[-1]}")
                shutil.copy2(src_image_path, dst_image_path)

if __name__ == "__main__":
    # Process directories based on the output directory
    process_directories(D2_TEST_OUT_DIR, D2_TEST_SC_DIR, MUTATIONS)