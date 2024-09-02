import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm  # For progress bar
import time  # For time measurement

# Input Parameters
SEED = 42
DATA_DIR = '/home/priyansh/Downloads/datasets/d1_classify/balanced'
MODEL_DIR = '/home/priyansh/Downloads/code/weights/d1_classification_balanced'
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'last.pth')
BATCH_SIZE = 128
NUM_EPOCHS = 1

# Set seed for reproducibility
torch.manual_seed(SEED)

# Image transformations for all phases (train, val, test)
data_transforms = transforms.Compose([
    transforms.Resize(224),  # Resize to the model's input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets from the corresponding directories
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=data_transforms),
    'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=data_transforms),
    'test': datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=data_transforms)
}

# Data loaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

# Load pre-trained ResNet model
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

# Modify the final fully connected layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification (2 classes)

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0.0
best_epoch = 0

# Load checkpoint if exists
start_epoch = 0
if os.path.isfile(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    best_epoch = checkpoint['best_epoch']
else:
    print("No checkpoint found, starting from pre-trained model")

# Training loop
# for epoch in range(start_epoch, NUM_EPOCHS):
#     print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
#     print('-' * 20)

#     start_time = time.time()  # Start timing

#     # Each epoch has a training and validation phase
#     for phase in ['train', 'val']:
#         if phase == 'train':
#             model.train()
#         else:
#             model.eval()

#         running_loss = 0.0
#         running_corrects = 0

#         # Use tqdm for progress bar
#         for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Phase', unit='batch'):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward pass
#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)

#                 # Backward pass and optimize only if in training phase
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()

#             # Statistics
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / len(dataloaders[phase].dataset)
#         epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

#         print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#         # Deep copy the model if it has better accuracy
#         if phase == 'val' and epoch_acc > best_acc:
#             best_acc = epoch_acc
#             best_epoch = epoch
#             torch.save({
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'best_acc': best_acc,
#                 'best_epoch': best_epoch
#             }, os.path.join(MODEL_DIR, 'best.pth'))
#             print(f'Best model saved with accuracy: {best_acc:.4f}')

#     epoch_time = time.time() - start_time  # End timing
#     print(f'Epoch {epoch+1} completed in {epoch_time:.2f} seconds\n')

# # Save the last model weights
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'epoch': NUM_EPOCHS - 1,
#     'best_acc': best_acc,
#     'best_epoch': best_epoch
# }, os.path.join(MODEL_DIR, 'last.pth'))
# print(f'Last model saved at epoch {NUM_EPOCHS}')

# Function to calculate accuracy on the test dataset
def test_model(model, dataloader, device):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / len(dataloader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')

# Load and evaluate the best model
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best.pth'))['model_state_dict'])
print('Evaluating the best model:')
test_model(model, dataloaders['test'], device)

# Load and evaluate the last model
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'last.pth'))['model_state_dict'])
print('Evaluating the last model:')
test_model(model, dataloaders['test'], device)