import torch, os, sys, time
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import *
torch.manual_seed(SEED)

class ModelTrainer:
    def __init__(self, data_dir, device, model_dir, checkpoint_path, batch_size, num_epochs, lr, momentum, mode, cancer_cells, prediction_only=False, class_weights=None, transform=None):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.momentum = momentum
        self.mode = mode
        self.cancer_cells = cancer_cells
        self.class_weights = class_weights
        self.prediction_only = prediction_only
        self.device = torch.device(device)
        self.best_acc = 0.0
        self.start_epoch = 0
        self.training_stats = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.cell_type_correct = defaultdict(int)
        self.cell_type_total = defaultdict(int)
        self.class_correct = {'cancer': 0, 'non_cancer': 0}
        self.class_total = {'cancer': 0, 'non_cancer': 0}

        self._setup(transform)

    def _setup(self, transform):
        os.makedirs(self.model_dir, exist_ok=True)
        self._initialize_transform(transform)
        self._initialize_model()
        self._initialize_optimizer()
        if not self.prediction_only:
            self._initialize_data_loaders()
        self._load_checkpoint()

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

    def _initialize_optimizer(self):
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
            self.training_stats = checkpoint['training_stats']
            self.cell_type_correct = checkpoint['cell_type_correct']
            self.cell_type_total = checkpoint['cell_type_total']
            self.class_correct = checkpoint['class_correct']
            self.class_total = checkpoint['class_total']
        else:
            print("No checkpoint found, starting from scratch")

    def _save_checkpoint(self, name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.num_epochs - 1,
            'best_acc': self.best_acc,
            'training_stats': self.training_stats,
            'cell_type_correct': self.cell_type_correct,
            'cell_type_total': self.cell_type_total,
            'class_correct': self.class_correct,
            'class_total': self.class_total
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
                    original_labels = labels  # Keep the original labels for accuracy tracking

                    # Remap labels to binary for training
                    binary_labels = torch.tensor([1 if self.dataloaders['train'].dataset.classes[label] in self.cancer_cells else 0 for label in labels]).to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, binary_labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                        preds = torch.max(outputs, 1)[1]  # Get the predictions

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == binary_labels.data)

                        # Update cell-type-wise and class-wise accuracy
                        for i, label in enumerate(original_labels):
                            cell_type = self.dataloaders['train'].dataset.classes[label]
                            if cell_type in self.cancer_cells:
                                self.class_correct['cancer'] += (preds[i] == 1).item()  # Correct if prediction is 1
                                self.class_total['cancer'] += 1
                            else:
                                self.class_correct['non_cancer'] += (preds[i] == 0).item()  # Correct if prediction is 0
                                self.class_total['non_cancer'] += 1

                            self.cell_type_correct[cell_type] += (preds[i] == binary_labels[i]).item()
                            self.cell_type_total[cell_type] += 1

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Display cell-type-wise accuracy
                print(f'{phase.capitalize()} Cell-type-wise Accuracy:')
                for cell_type in self.cell_type_correct:
                    acc = self.cell_type_correct[cell_type] / self.cell_type_total[cell_type] if self.cell_type_total[cell_type] > 0 else 0
                    print(f'{cell_type}: {acc:.4f}')

                # Display class-wise accuracy (cancer vs non-cancer)
                cancer_acc = self.class_correct['cancer'] / self.class_total['cancer'] if self.class_total['cancer'] > 0 else 0
                non_cancer_acc = self.class_correct['non_cancer'] / self.class_total['non_cancer'] if self.class_total['non_cancer'] > 0 else 0
                print(f'Cancer Accuracy: {cancer_acc:.4f}, Non-cancer Accuracy: {non_cancer_acc:.4f}')

                self.training_stats[f'{phase}_loss'].append(epoch_loss)
                self.training_stats[f'{phase}_acc'].append(epoch_acc)

                if epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self._save_checkpoint('best.pth')

            epoch_time = time.time() - start_time
            print(f'Epoch {epoch + 1} completed in {epoch_time:.2f} seconds\n')

        self._save_checkpoint('last.pth')

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
    trainer = ModelTrainer(D1_DATA_DIR, DEVICE, MODEL_DIR, CHK_PTH, BATCH_SIZE, NUM_EPOCHS, LR, MOMENTUM, MODE, PREDICTION_ONLY, CANCER_CELLS, CLASS_WEIGHTS)
    trainer.train()