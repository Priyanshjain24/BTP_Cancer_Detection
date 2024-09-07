import torch
from torchvision import models
from sklearn.cluster import KMeans
import numpy as np

class KMeansPredictor:
    def __init__(self, device):
        self.device = torch.device(device)
        self._initialize_resnet()

    def _initialize_resnet(self):
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.resnet = self.resnet.to(self.device)
        self.resnet.eval()

    def predict(self, data_loader):
        features = []
        img_names = []

        with torch.no_grad():
            for imgs, names in data_loader:
                imgs = imgs.to(self.device)
                outputs = self.resnet(imgs).squeeze()
                features.append(outputs.cpu().numpy())
                img_names.extend(names)

        features = np.vstack(features)
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(features)

        return img_names, labels