import os, shutil, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataloader import SingleClassImageDataset
from torch.utils.data import DataLoader
from v5.classification import ModelTrainer
from torchvision import transforms
from constants import *

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainer = ModelTrainer(D2_TEST_OUT_DIR, DEVICE, MODEL_DIR, CHK_PTH, BATCH_SIZE, NUM_EPOCHS, LR, MOMENTUM, WEIGHT_DECAY, MODEL, True, CLASS_WEIGHTS, DROPOUT, None, PATIENCE, DELTA, NUM_CLASSES, FREEZE_LAYERS)
    
    for disease_folder in MUTATIONS:
        os.makedirs(os.path.join(D2_TEST_PREDS_DIR, disease_folder), exist_ok=True)

    # Iterate through disease folders
    for disease_folder in MUTATIONS:
        disease_path = os.path.join(D2_TEST_SC_DIR, 'val', disease_folder)

        if os.path.isdir(disease_path):
            # Iterate through patient folders
            dataset = SingleClassImageDataset(disease_path, transform=transform)
            data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

            labels, probs = trainer.predict(data_loader)
            img_names = dataset.img_names

            for img_name, label in zip(img_names, labels):
                src_path = os.path.join(disease_path, img_name)
                dst_dir = MUTATIONS[label]
                shutil.copy(src_path, os.path.join(D2_TEST_PREDS_DIR, dst_dir, img_name))