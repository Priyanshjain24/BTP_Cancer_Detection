import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch import nn, optim

# --- USER PARAMETERS (edit these) ---
DATA1 = "/home/Drivehd2tb/garima/datasets/mod_data/d1_classify/complete_balanced"
DATA2 = "/home/Drivehd2tb/garima/datasets/mod_data/d2_classify/sc"
NUM_CLASSES1 = 2
NUM_CLASSES2 = 4
MODEL_NAME = "resnet18"
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-5
VAL_SPLIT = 0.2  # fraction for validation
OUTPUT_DIR = "/home/Drivehd2tb/garima/code/weights_new/joint/v1"

# --- SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

# --- TRANSFORMS ---
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# --- DATA SPLIT FUNCTION ---
def make_loaders(path, tf_train, tf_val, split, bs):
    ds_train_full = datasets.ImageFolder(path, transform=tf_train)
    ds_val_full = datasets.ImageFolder(path, transform=tf_val)
    n = len(ds_train_full)
    idx = list(range(n))
    np.random.shuffle(idx)
    split_at = int((1 - split) * n)
    train_idx, val_idx = idx[:split_at], idx[split_at:]
    train_ds = Subset(ds_train_full, train_idx)
    val_ds = Subset(ds_val_full, val_idx)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    )

train_loader1, val_loader1 = make_loaders(DATA1, train_tf, val_tf, VAL_SPLIT, BATCH_SIZE)
train_loader2, val_loader2 = make_loaders(DATA2, train_tf, val_tf, VAL_SPLIT, BATCH_SIZE)

# --- MODEL DEFINITION ---
def get_shared_model(backbone, nc1, nc2):
    base = getattr(models, backbone)(weights="DEFAULT")
    feat_dim = base.fc.in_features
    base.fc = nn.Identity()
    head1 = nn.Linear(feat_dim, nc1)
    head2 = nn.Linear(feat_dim, nc2)
    return base, head1, head2

encoder, head1, head2 = get_shared_model(MODEL_NAME, NUM_CLASSES1, NUM_CLASSES2)
encoder, head1, head2 = encoder.to(device), head1.to(device), head2.to(device)

optimizer = optim.AdamW(
    list(encoder.parameters()) + list(head1.parameters()) + list(head2.parameters()), lr=LR
)
criterion = nn.CrossEntropyLoss()

# --- STATS ---
stats = {
    "train_loss1": [], "val_loss1": [],
    "train_loss2": [], "val_loss2": [],
    "train_acc1":  [], "val_acc1":  [],
    "train_acc2":  [], "val_acc2":  []
}

# --- TRAIN & VALIDATION LOOP ---
for epoch in range(EPOCHS):
    encoder.train(); head1.train(); head2.train()

    # Alternate training
    if epoch % 2 == 0:
        # Train on D1
        running_loss, running_correct = 0.0, 0
        for x, y in tqdm(train_loader1, desc=f"[Epoch {epoch+1}] Train D1"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            feats = encoder(x)
            out = head1(feats)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
        stats["train_loss1"].append(running_loss / len(train_loader1.dataset))
        stats["train_acc1"].append(running_correct / len(train_loader1.dataset))
        stats["train_loss2"].append(None)
        stats["train_acc2"].append(None)

    else:
        # Train on D2
        running_loss, running_correct = 0.0, 0
        for x, y in tqdm(train_loader2, desc=f"[Epoch {epoch+1}] Train D2"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            feats = encoder(x)
            out = head2(feats)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
        stats["train_loss2"].append(running_loss / len(train_loader2.dataset))
        stats["train_acc2"].append(running_correct / len(train_loader2.dataset))
        stats["train_loss1"].append(None)
        stats["train_acc1"].append(None)

    # Validation for both
    encoder.eval(); head1.eval(); head2.eval()
    with torch.no_grad():
        # D1 val
        vloss, vcorrect = 0.0, 0
        for x, y in val_loader1:
            x, y = x.to(device), y.to(device)
            out = head1(encoder(x))
            vloss += criterion(out, y).item() * x.size(0)
            vcorrect += (out.argmax(1) == y).sum().item()
        stats["val_loss1"].append(vloss / len(val_loader1.dataset))
        stats["val_acc1"].append(vcorrect / len(val_loader1.dataset))

        # D2 val
        vloss, vcorrect = 0.0, 0
        for x, y in val_loader2:
            x, y = x.to(device), y.to(device)
            out = head2(encoder(x))
            vloss += criterion(out, y).item() * x.size(0)
            vcorrect += (out.argmax(1) == y).sum().item()
        stats["val_loss2"].append(vloss / len(val_loader2.dataset))
        stats["val_acc2"].append(vcorrect / len(val_loader2.dataset))

    # Logging
    if stats['train_acc1'][-1] is not None:
        print(f"[Epoch {epoch+1}] D1 → Train Acc: {stats['train_acc1'][-1]:.4f}  Val Acc: {stats['val_acc1'][-1]:.4f}")
    if stats['train_acc2'][-1] is not None:
        print(f"[Epoch {epoch+1}] D2 → Train Acc: {stats['train_acc2'][-1]:.4f}  Val Acc: {stats['val_acc2'][-1]:.4f}")


# --- SAVE MODEL ---
torch.save({
    "encoder": encoder.state_dict(),
    "head1": head1.state_dict(),
    "head2": head2.state_dict()
}, os.path.join(OUTPUT_DIR, "joint_model.pth"))

# --- PLOT & EVALUATE ---
def plot_and_evaluate():
    # Confusion matrix and accuracy
    def eval_cm(head, loader, name):
        head.eval(); encoder.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = head(encoder(x)).argmax(1)
                preds.extend(out.cpu().numpy())
                trues.extend(y.cpu().numpy())
        cm = confusion_matrix(trues, preds)
        acc = (np.array(preds) == np.array(trues)).mean()
        np.savetxt(os.path.join(OUTPUT_DIR, f"cm_{name}.csv"), cm, delimiter=",")
        print(f"{name.upper()} final accuracy: {acc:.4f}")

    plot_items = [("loss1", "Loss D1"), ("acc1", "Acc D1"),
                  ("loss2", "Loss D2"), ("acc2", "Acc D2")]
    for key, title in plot_items:
        tv = [v for v in stats[f"train_{key}"] if v is not None]
        vv = stats[f"val_{key}"]
        plt.figure()
        plt.plot(tv, label="Train")
        plt.plot(vv, label="Val")
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{key}.png"))
        plt.close()

    eval_cm(head1, val_loader1, "d1")
    eval_cm(head2, val_loader2, "d2")

plot_and_evaluate()
print("Done.")