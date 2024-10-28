import os, shutil, random, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import *
random.seed(SEED)

def split_train_val(base_dir, mutations, split_ratio):

    # Create train and val directories
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate over each class folder
    for class_folder in mutations:
        class_folder_path = os.path.join(base_dir, class_folder)
        
        if os.path.isdir(class_folder_path):  # Ensure it's a directory
            # Create class directories in train and val folders
            train_class_dir = os.path.join(train_dir, class_folder)
            val_class_dir = os.path.join(val_dir, class_folder)
            
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
            
            # Get list of all files in the class folder
            files = os.listdir(class_folder_path)
            random.shuffle(files)  # Shuffle the files randomly
            
            # Split files into train and val sets
            train_count = int(len(files) * split_ratio[0])
            
            # Copy files to respective directories
            for i, file in enumerate(files):
                src_path = os.path.join(class_folder_path, file)
                if i <= train_count:
                    dest_path = os.path.join(train_class_dir, file)
                else:
                    dest_path = os.path.join(val_class_dir, file)
                
                shutil.copy(src_path, dest_path)
            
            # Optionally remove the original class folder
            shutil.rmtree(class_folder_path)

    print("Data successfully split into train and val directories.")
    
if __name__ == "__main__":
    split_train_val(D2_TEST_SC_DIR, MUTATIONS, SPLIT_RATIO)
