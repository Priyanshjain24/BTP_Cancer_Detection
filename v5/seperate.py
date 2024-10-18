import os, shutil, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataloader import SingleClassImageDataset
from torch.utils.data import DataLoader
from v5.classification import ModelTrainer
from torchvision import transforms
import pandas as pd
from constants import *

def process_patient_folder(patient_dir, output_patient_dir, trainer=None, transform=None):
    dataset = SingleClassImageDataset(patient_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    labels, probs = trainer.predict(data_loader)
    img_names = dataset.img_names

    os.makedirs(os.path.join(output_patient_dir, 'cancer'), exist_ok=True)
    os.makedirs(os.path.join(output_patient_dir, 'non_cancer'), exist_ok=True)
    
    probs = pd.DataFrame(probs, columns=['cancer', 'non_cancer'])
    probs = pd.concat([pd.Series(img_names, name='name'), probs], axis=1)
    probs.to_csv(os.path.join(output_patient_dir, 'probs.csv'), index=False)

    for img_name, label in zip(img_names, labels):
        src_path = os.path.join(patient_dir, img_name)
        dst_dir = 'cancer' if label == 0 else 'non_cancer'
        shutil.copy(src_path, os.path.join(output_patient_dir, dst_dir, img_name))

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainer = ModelTrainer(D1_DATA_DIR, DEVICE, MODEL_DIR, CHK_PTH, BATCH_SIZE, NUM_EPOCHS, LR, MOMENTUM, WEIGHT_DECAY, MODEL, True, CLASS_WEIGHTS, DROPOUT, None, PATIENCE, DELTA, NUM_CLASSES, FREEZE_LAYERS)

    # Iterate through disease folders
    for disease_folder in os.listdir(D2_TEST_INP_DIR):
        disease_path = os.path.join(D2_TEST_INP_DIR, disease_folder)

        if os.path.isdir(disease_path):
            # Iterate through patient folders
            for patient_folder in os.listdir(disease_path):
                patient_path = os.path.join(disease_path, patient_folder)

                if os.path.isdir(patient_path):
                    output_patient_dir = os.path.join(D2_TEST_OUT_DIR, disease_folder, patient_folder)

                    # Process the patient folder and separate images into cancer/non_cancer
                    process_patient_folder(patient_path, output_patient_dir, trainer, transform)
