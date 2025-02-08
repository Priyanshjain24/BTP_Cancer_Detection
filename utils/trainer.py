import os
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class TrainerConfig:
    def __init__(self, data_dir, model, batch_size, learning_rate, weight_decay, num_workers=4, model_dir="checkpoints"):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.model = model
        self.model_dir = model_dir

        self.data_transforms = self._initialize_transform()
        self.dataloaders, self.class_names = self._initialize_data_loaders()
        self.optimizer, self.criterion = self._setup_optimizer()

    def _initialize_transform(self):
        """Set up default data transformations."""
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _initialize_data_loaders(self):
        """Set up PyTorch data loaders."""
        image_datasets = {
            'train': datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.data_transforms),
            'val': datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.data_transforms)
        }
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True),
            'val': DataLoader(image_datasets['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        }
        return dataloaders, image_datasets['train'].classes

    def _setup_optimizer(self):
        """Set up optimizer and loss function."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        return optimizer, criterion

    def write_csv(self, filename, conf_matrix):
        """Save confusion matrix to CSV file."""
        os.makedirs(self.model_dir, exist_ok=True)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=self.class_names, columns=self.class_names)
        conf_matrix_df.to_csv(os.path.join(self.model_dir, filename))
