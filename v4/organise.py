import os, shutil, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import *

def process_directories(input_dir, output_dir, mil=False):
    # Define the mutations
    mutations = ['CBFB_MYH11', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']

    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Copy all mutation folders from input_dir to output_dir, except the control folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for mutation in mutations:
        mutation_input_folder = os.path.join(input_dir, mutation)
        mutation_output_folder = os.path.join(output_dir, mutation)
        
        if not os.path.exists(mutation_input_folder):
            continue
        
        os.makedirs(mutation_output_folder, exist_ok=True)

        for patient in os.listdir(mutation_input_folder):
            patient_input_folder = os.path.join(mutation_input_folder, patient)
            patient_output_folder = os.path.join(mutation_output_folder, patient)
            cancer_input_folder = os.path.join(patient_input_folder, 'cancer')
            
            if os.path.exists(cancer_input_folder):
                for image in os.listdir(cancer_input_folder):
                    src_image_path = os.path.join(cancer_input_folder, image)
                    if mil:
                        os.makedirs(patient_output_folder, exist_ok=True)
                        dst_image_path = os.path.join(patient_output_folder, image)
                    else:
                        dst_image_path = os.path.join(mutation_output_folder, image)
                    shutil.copy2(src_image_path, dst_image_path)

if __name__ == "__main__":
    # Set MIL to True or False as needed
    MIL = False  # Change to True if needed
    output_directory = D2_TEST_MIL_DIR if MIL else D2_TEST_SC_DIR

    # Process directories based on the MIL flag
    process_directories(D2_TEST_OUT_DIR, output_directory, MIL)