import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time
from PIL import Image
from constants import *
torch.manual_seed(SEED)

class ModelTrainer:
    def __init__(self, data_dir, device, model_dir, checkpoint_path, batch_size, num_epochs, lr, momentum, mode, prediction_only, class_weights=None, transform=None):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum = momentum
        self.mode = mode
        self.prediction_only = prediction_only
        self.class_weights = class_weights
        self.device = torch.device(device)
        self.best_acc = 0.0
        self.start_epoch = 0
        self.training_stats = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        self._setup(transform)


    def _setup(self, transform):

        os.makedirs(self.model_dir, exist_ok=True)

        self._intialize_transform(transform)
        self._initialize_model()
        self._load_checkpoint()

        if not self.prediction_only:
            self._initialize_data_loaders()  # Skip data loader initialization for prediction
            self._initialize_optimizer()

    def _initialize_model(self):
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        num_ftrs = self.model.fc.in_features

        if self.mode == 'binary':
            self.model.fc = nn.Linear(num_ftrs, 2)
            if self.class_weights is not None:
                weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif self.mode == 'anomaly':
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 1),
                nn.Sigmoid()
            )
            self.criterion = nn.BCELoss()

        self.model = self.model.to(self.device)

    def _intialize_transform(self, transform):
        if transform is None:
            self.data_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.data_transforms = transform

    def _initialize_data_loaders(self):
        image_datasets = {
            'train': datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.data_transforms),
            'val': datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.data_transforms),
            'test': datasets.ImageFolder(os.path.join(self.data_dir, 'test'), transform=self.data_transforms)
        }

        if self.mode == 'anomaly':
            majority_class_idx = 1
            image_datasets['train'].samples = [s for s in image_datasets['train'].samples if s[1] == majority_class_idx]

        self.dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(image_datasets['val'], batch_size=self.batch_size, shuffle=False, num_workers=4),
            'test': DataLoader(image_datasets['test'], batch_size=self.batch_size, shuffle=False, num_workers=4)
        }

    def _initialize_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    def _load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if not self.prediction_only:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_acc = checkpoint['best_acc']
                self.training_stats = checkpoint['training_stats']
        else:
            print("No checkpoint found, starting from scratch")

    def _save_checkpoint(self, name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.num_epochs - 1,
            'best_acc': self.best_acc,
            'training_stats': self.training_stats
        }, os.path.join(self.model_dir, name))

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 20)

            start_time = time.time()

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(self.dataloaders[phase], desc=f'{phase.capitalize()} Phase', unit='batch'):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    if self.mode == 'anomaly':
                        labels = labels.float().unsqueeze(1)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                        preds = torch.max(outputs, 1)[1] if self.mode == 'binary' else (outputs > 0.5).float()
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                self.training_stats[f'{phase}_loss'].append(epoch_loss)
                self.training_stats[f'{phase}_acc'].append(epoch_acc)

                if epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self._save_checkpoint('best.pth')

            epoch_time = time.time() - start_time
            print(f'Epoch {epoch + 1} completed in {epoch_time:.2f} seconds\n')

        self._save_checkpoint('last.pth')

    def test(self, checkpoint='best.pth'):
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, checkpoint))['model_state_dict'])
        self.model.eval()

        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in self.dataloaders['test']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                if self.mode == 'anomaly':
                    labels = labels.float().unsqueeze(1)

                outputs = self.model(inputs)
                preds = torch.max(outputs, 1)[1] if self.mode == 'binary' else (outputs > 0.5).float()
                running_corrects += torch.sum(preds == labels.data)

        accuracy = running_corrects.double() / len(self.dataloaders['test'].dataset)
        print(f'Test Accuracy: {accuracy:.4f}')

    def predict(self, data_loader):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for inputs, _ in tqdm(data_loader, desc="Predicting", unit="batch"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = torch.max(outputs, 1)[1] if self.mode == 'binary' else (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
        return all_preds

# Starting Training
if __name__ == "__main__":
    trainer = ModelTrainer(D1_DATA_DIR, DEVICE, MODEL_DIR, CHK_PTH, BATCH_SIZE, NUM_EPOCHS, LR, MOMENTUM, MODE, CLASS_WEIGHTS)
    trainer.train()
    trainer.test()