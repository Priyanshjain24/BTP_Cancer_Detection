import torch, time, tqdm, os, sys, argparse, json
torch.manual_seed(42)
torch.hub.set_dir('/home/Drivehd2tb/garima/cache')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.models import ModelManager
from utils.trainer import TrainerConfig
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a deep learning model.")
parser.add_argument("--model", type=str, default="RESNET18", help="Model name (e.g., RESNET18, VGG16)")
parser.add_argument("--num_classes", type=int, required=True, help="Number of output classes")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
parser.add_argument("--delta", type=float, default=0.01, help="Minimum loss improvement for early stopping")
parser.add_argument("--model_dir", type=str, required=True, help="Directory to save model weights")
parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs if available")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to use (cpu or cuda)")
parser.add_argument("--k_folds", type=int, default=4, help="Number of folds for cross-validation")

args = parser.parse_args()

# Device setup: If not provided, assign automatically
DEVICE = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load dataset
base_dataset = datasets.ImageFolder(root=args.data_dir)
labels = [sample[1] for sample in base_dataset.imgs]  # Extract labels for stratified splitting

# K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

# Track overall k-fold accuracy stats
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(base_dataset)), labels)):
    print(f"\n========== Fold {fold + 1}/{args.k_folds} ==========")

    # Create folder for this fold
    fold_dir = os.path.join(args.model_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Initialize model and training utilities for this fold
    model_manager = ModelManager(
        model_name=args.model,
        num_classes=args.num_classes,
        device=DEVICE,
        checkpoint_path=os.path.join(fold_dir, "best.pth"),
        use_multi_gpu=args.multi_gpu,
        model_dir=fold_dir
    )

    trainer_config = TrainerConfig(
        None, model_manager.model, args.batch_size, args.lr, args.weight_decay, model_dir=fold_dir
    )

    # Create dataset subsets with different transforms
    train_dataset = datasets.ImageFolder(root=args.data_dir, transform=trainer_config.train_transforms)
    val_dataset = datasets.ImageFolder(root=args.data_dir, transform=trainer_config.val_transforms)

    trainer_config.class_names = train_dataset.classes
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=trainer_config.batch_size, shuffle=True, num_workers=trainer_config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=trainer_config.batch_size, shuffle=False, num_workers=trainer_config.num_workers, pin_memory=True)

    model_manager.optimizer = trainer_config.optimizer

    start_epoch, training_stats = model_manager.load_checkpoint(trainer_config.optimizer)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    if not training_stats:
        training_stats = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Training loop for the current fold
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print('-' * 20)

        start_time = time.time()

        for phase in ['train', 'val']:
            model_manager.model.train() if phase == 'train' else model_manager.model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []

            data_loader = train_loader if phase == 'train' else val_loader

            for inputs, labels in tqdm.tqdm(data_loader, desc=f'{phase.capitalize()} Phase', unit='batch'):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                trainer_config.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_manager.model(inputs)
                    loss = trainer_config.criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        trainer_config.optimizer.step()

                    preds = torch.argmax(outputs, dim=1)
                    running_loss += loss.item() * inputs.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

            # Calculate metrics
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
            conf_matrix = confusion_matrix(all_labels, all_preds)

            # Store stats
            training_stats[f'{phase}_loss'].append(epoch_loss)
            training_stats[f'{phase}_acc'].append(epoch_acc)

            print(f'{phase.capitalize()} Overall Accuracy: {epoch_acc:.4f}')
            class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
            for idx, class_name in enumerate(trainer_config.class_names):
                print(f'{phase.capitalize()} Accuracy for {class_name}: {class_accuracies[idx]:.4f}')

            # Validation loss tracking for early stopping
            if phase == 'val':
                if epoch_loss > best_val_loss + args.delta:
                    early_stopping_counter += 1
                    print(f"Early stopping counter: {early_stopping_counter}/{args.patience}")
                else:
                    early_stopping_counter = 0
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        model_manager.save_checkpoint(os.path.join(fold_dir, "best.pth"), epoch, training_stats)
                        trainer_config.write_csv(os.path.join(fold_dir, 'best_conf_matrix.csv'), conf_matrix)

        end_time = time.time()
        print(f"Epoch duration: {end_time - start_time:.2f} seconds")

        if early_stopping_counter >= args.patience:
            print("Early stopping triggered")
            break

    model_manager.save_checkpoint(os.path.join(fold_dir, "last.pth"), args.epochs, training_stats)
    trainer_config.write_csv(os.path.join(fold_dir, 'last_conf_matrix.csv'), conf_matrix)
    trainer_config.save_plots(training_stats)

    # Store final accuracy for this fold
    fold_accuracies.append(training_stats['val_acc'][-1])

# Save k-fold overall stats
stats_summary = {
    "mean_accuracy": np.mean(fold_accuracies),
    "std_accuracy": np.std(fold_accuracies),
    "min_accuracy": np.min(fold_accuracies),
    "max_accuracy": np.max(fold_accuracies),
    "fold_accuracies": fold_accuracies
}

with open(os.path.join(args.model_dir, "kfold_summary.json"), "w") as f:
    json.dump(stats_summary, f, indent=4)

print("\nK-Fold training complete!")