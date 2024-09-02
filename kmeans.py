import os
import shutil
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

class SingleClassImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.img_names[idx]
    
input_dir = "testdata_input"
output_dir = "testdata_output"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

dataset = SingleClassImageDataset(input_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)
resnet.eval()

features = []
img_names = []

with torch.no_grad():
    for imgs, names in data_loader:
        imgs = imgs.to(device)
        outputs = resnet(imgs).squeeze()
        features.append(outputs.cpu().numpy())
        img_names.extend(names)

features = np.vstack(features)

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(features)

os.makedirs(os.path.join(output_dir, 'cancer'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'non_cancer'), exist_ok=True)

for img_name, label in zip(img_names, labels):
        src_path = os.path.join(input_dir, img_name)
        dst_dir = f'class:{label}'
        shutil.copy(src_path, os.path.join(output_dir, dst_dir, img_name))