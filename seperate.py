import os
import shutil
from utils.kmeans import KMeansPredictor
from utils.dataloader import SingleClassImageDataset
from torch.utils.data import DataLoader
from classification import ModelTrainer
from torchvision import transforms
from constants import *

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = SingleClassImageDataset(D2_TEST_INP_DIR, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    if KMEANS:
        kmeans_predictor = KMeansPredictor(device="cuda")
        img_names, labels = kmeans_predictor.predict(data_loader)
    else:
        trainer = ModelTrainer(D2_TEST_INP_DIR, DEVICE, MODEL_DIR, CHK_PTH, BATCH_SIZE, NUM_EPOCHS, LR, MOMENTUM, MODE, PREDICTION_ONLY, CLASS_WEIGHTS)
        labels = trainer.predict(data_loader)
        img_names = dataset.img_names

    os.makedirs(os.path.join(D2_TEST_OUT_DIR, 'cancer'), exist_ok=True)
    os.makedirs(os.path.join(D2_TEST_OUT_DIR, 'non_cancer'), exist_ok=True)

    for img_name, label in zip(img_names, labels):
        src_path = os.path.join(D2_TEST_INP_DIR, img_name)
        dst_dir = 'cancer' if label == 0 else 'non_cancer'
        shutil.copy(src_path, os.path.join(D2_TEST_OUT_DIR, dst_dir, img_name))
