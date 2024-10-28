import torch, os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import *
torch.manual_seed(SEED)
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

class ModelTrainer:
    def __init__(self, data_dir=None, device='cpu', model_dir=None, checkpoint_path=None, batch_size=16, num_epochs=100, lr=1e-3, momentum=0.9, weight_decay=1e-4, model_name='RESNET18', prediction_only=False, class_weights=None, dropout_p=0.5, transform=None, patience=10, delta=0.01, num_classes=2, freeze_layers=False):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.prediction_only = prediction_only
        self.device = torch.device(device)
        self.model_name = model_name
        self.dropout_p = dropout_p
        self.start_epoch = 0
        self.patience = patience
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.delta = delta
        self.num_classes = num_classes
        self.class_names = []

        # Initialize training stats
        self.training_stats = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        self._setup(transform, freeze_layers)

    def _setup(self, transform, freeze_layers):
        os.makedirs(self.model_dir, exist_ok=True)
        self._initialize_transform(transform)
        self._initialize_model(freeze_layers)
        self._initialize_optimizer()
        if not self.prediction_only:
            self._initialize_data_loaders()
        self._load_checkpoint()

    def _initialize_model(self, freeze_layers):
        if self.model_name == 'RESNET18':
            self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        elif self.model_name == 'RESNET34':
            self.model = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        elif self.model_name == 'VGG11':
            self.model = models.vgg11_bn(weights='VGG11_BN_Weights.DEFAULT')
        elif self.model_name == 'VGG13':
            self.model = models.vgg13_bn(weights='VGG13_BN_Weights.DEFAULT')
        elif self.model_name == 'VGG16':
            self.model = models.vgg16_bn(weights='VGG16_BN_Weights.DEFAULT')
        elif self.model_name == 'VGG19':
            self.model = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT')
        elif self.model_name == 'RESNET50':
            self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        elif self.model_name == 'CONVNEXT_TINY':
            self.model = models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.DEFAULT')
        elif self.model_name == 'REGNET':
            self.model = models.regnet_y_400mf(weights='RegNet_Y_400MF_Weights.DEFAULT')
        elif self.model_name == 'SWIN_TRANSFORMER_TINY':
            self.model = models.swin_t(weights='Swin_T_Weights.DEFAULT')
        elif self.model_name == 'EFFICIENT_NET':
            self.model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        for param in self.model.parameters():
            param.requires_grad = not freeze_layers

        num_ftrs = self.model.fc.in_features if hasattr(self.model, 'fc') else self.model.classifier[-1].in_features
        layer = nn.Sequential(
            nn.Dropout(self.dropout_p),
            nn.Linear(num_ftrs, self.num_classes))  # number of output classes
        
         # Only set the final classification layer to be trainable
        if hasattr(self.model, 'fc'):
            self.model.fc = layer
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            self.model.classifier[-1] = layer
            for param in self.model.classifier[-1].parameters():
                param.requires_grad = True

        weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device) if self.class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.2)
        self.model = self.model.to(self.device)

    def _initialize_transform(self, transform):
        if transform is None:
            self.data_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.data_transforms = transform

    def _initialize_data_loaders(self):
        image_datasets = {
            'train': datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.data_transforms),
            'val': datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.data_transforms)
        }
        self.dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(image_datasets['val'], batch_size=self.batch_size, shuffle=False, num_workers=4)
        }
        self.class_names = image_datasets['train'].classes

    def _initialize_optimizer(self):
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def _load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.training_stats = checkpoint['training_stats']
        else:
            print("No checkpoint found, starting from scratch")

    def _save_checkpoint(self, name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.num_epochs - 1,
            'training_stats': self.training_stats,
        }, os.path.join(self.model_dir, name))

    def _write_csv(self, filename, conf_matrix):
        """Save only the confusion matrix to a CSV file."""
        conf_matrix_df = pd.DataFrame(conf_matrix, index=self.class_names, columns=self.class_names)
        conf_matrix_df.to_csv(filename, mode='w')

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
                correct_predictions = {class_name: 0 for class_name in self.class_names}
                total = {class_name: 0 for class_name in self.class_names}
                all_labels = []
                all_preds = []

                for inputs, labels in tqdm(self.dataloaders[phase], desc=f'{phase.capitalize()} Phase', unit='batch'):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                        preds = torch.max(outputs, 1)[1]
                        running_loss += loss.item() * inputs.size(0)

                        # Collect predictions and labels
                        all_labels.extend(labels.cpu().numpy())
                        all_preds.extend(preds.cpu().numpy())

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()

                # Calculate confusion matrix
                conf_matrix = confusion_matrix(all_labels, all_preds)
                
                self.training_stats[f'{phase}_loss'].append(epoch_loss)
                self.training_stats[f'{phase}_acc'].append(epoch_acc)
                print(f'{phase.capitalize()} Overall Accuracy: {epoch_acc:.4f}')

                # Calculate class-wise accuracies
                class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
                for idx, class_name in enumerate(self.class_names):
                    print(f'{phase.capitalize()} Accuracy for {class_name}: {class_accuracies[idx]:.4f}')

                if phase == 'val':
                    if epoch_loss > self.best_val_loss + self.delta:
                        self.early_stopping_counter += 1
                        print(f"Early stopping counter: {self.early_stopping_counter}/{self.patience}")
                    else:
                        self.early_stopping_counter = 0
                        if epoch_loss < self.best_val_loss:
                            self.best_val_loss = epoch_loss
                            self._save_checkpoint("best.pth")
                            self._write_csv(os.path.join(self.model_dir, 'best_conf_matrix.csv'), conf_matrix)

            end_time = time.time()
            print(f"Epoch duration: {end_time - start_time:.2f} seconds")

            if self.early_stopping_counter >= self.patience:
                print("Early stopping triggered")
                break

        self._save_checkpoint("last.pth")
        self._write_csv(os.path.join(self.model_dir, 'last_conf_matrix.csv'), conf_matrix)

    def predict(self, data_loader):
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(data_loader, desc="Predicting", unit="batch"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                # Get the probabilities using softmax
                probs = F.softmax(outputs, dim=1)
                # Get the predicted classes
                preds = torch.max(outputs, 1)[1]

                # Append predictions and probabilities
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return all_preds, all_probs

# Starting Training
if __name__ == "__main__":
    trainer = ModelTrainer(D2_TEST_SC_DIR, DEVICE, MODEL_DIR, CHK_PTH, BATCH_SIZE, NUM_EPOCHS, LR, MOMENTUM, WEIGHT_DECAY, MODEL, PREDICTION_ONLY, CLASS_WEIGHTS, DROPOUT, None, PATIENCE, DELTA, NUM_CLASSES, FREEZE_LAYERS)
    trainer.train()