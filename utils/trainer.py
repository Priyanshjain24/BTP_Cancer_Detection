import os
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class TrainerConfig:
    def __init__(self, data_dir, model, batch_size, learning_rate, weight_decay, num_workers=4, model_dir="checkpoints"):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.model = model
        self.model_dir = model_dir

        self.train_transforms, self.val_transforms = self._initialize_transform()
        if self.data_dir:
            self.dataloaders, self.class_names = self._initialize_data_loaders()
        self.optimizer, self.criterion = self._setup_optimizer()

    def _initialize_transform(self):
        """Set up default data transformations with augmentations."""
        train_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((10, 30)),
            transforms.ToTensor(),
            transforms.Normalize([0.815, 0.748, 0.86], [0.041, 0.067, 0.021])
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.815, 0.748, 0.86], [0.041, 0.067, 0.021])
        ])

        return train_transforms, val_transforms

    def _initialize_data_loaders(self):
        """Set up PyTorch data loaders."""
        image_datasets = {
            'train': datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.train_transforms),
            'val': datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.val_transforms)
        }
        
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True),
            'val': DataLoader(image_datasets['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        }
        return dataloaders, image_datasets['train'].classes

    def _setup_optimizer(self):
        """Set up optimizer and loss function."""
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        return optimizer, criterion

    def write_csv(self, filename, conf_matrix):
        """Save confusion matrix to CSV file."""
        conf_matrix_df = pd.DataFrame(conf_matrix, index=self.class_names, columns=self.class_names)
        conf_matrix_df.to_csv(os.path.join(self.model_dir, filename))

    def save_plots(self, stats):
        os.makedirs(self.model_dir, exist_ok=True)
        epochs = range(1, len(stats['train_loss']) + 1)
        
        plt.figure()
        plt.plot(epochs, stats['train_loss'], label='Train Loss')
        plt.plot(epochs, stats['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training & Validation Loss')
        plt.savefig(os.path.join(self.model_dir, 'loss_plot.png'))
        plt.close()
        
        plt.figure()
        plt.plot(epochs, stats['train_acc'], label='Train Accuracy')
        plt.plot(epochs, stats['val_acc'], label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training & Validation Accuracy')
        plt.savefig(os.path.join(self.model_dir, 'accuracy_plot.png'))
        plt.close()