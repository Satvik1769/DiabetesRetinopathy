import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import cv2
import time
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, cohen_kappa_score, roc_auc_score,
                             precision_score, recall_score, f1_score, confusion_matrix)

# ----- CONFIG -----
HDF5_PATH = './data/eyepacs.h5'
BATCH_SIZE = 32
IMAGE_SIZE = 728
EPOCHS = 10
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----- IMAGE PREPROCESSING -----
def resize_with_padding(img, desired_size=728):
    old_size = img.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def preprocess_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    img = img[y:y+h, x:x+w]
    img = resize_with_padding(img, IMAGE_SIZE)
    return Image.fromarray(img)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])
# ----- DATASET CLASS FOR HDF5 -----
class HDF5Dataset(Dataset):
    def __init__(self, h5_path, split, transform=None):
        self.file = h5py.File(h5_path, 'r')
        self.images = self.file[f"{split}/images"]
        self.labels = self.file[f"{split}/labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float)

# ----- MODEL -----
class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.fc = nn.Linear(self.backbone.classifier[1].in_features, 1)
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)

# ----- QWK METRIC -----
def quadratic_weighted_kappa(y_pred, y_true):
    y_pred = np.clip(np.round(y_pred), 0, 4).astype(int)
    y_true = np.array(y_true).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# ----- IOU METRIC -----
def iou_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    union = np.where(union == 0, 1, union)
    iou = intersection / union
    return np.mean(iou)

# ----- TRAINING LOOP -----
def train():
    model = DRModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_dataset = HDF5Dataset(HDF5_PATH, 'train', transform=transform)
    val_dataset = HDF5Dataset(HDF5_PATH, 'val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(EPOCHS):
        start_time = time.time()

        model.train()
        total_loss = 0
        train_preds, train_labels = [], []

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            imgs, labels = imgs.to(DEVICE), labels.unsqueeze(1).to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_preds.extend(preds.detach().cpu().numpy().squeeze())
            train_labels.extend(labels.cpu().numpy().squeeze())

        train_preds_bin = np.round(train_preds)
        train_labels_bin = np.round(train_labels)
        train_acc = accuracy_score(train_labels_bin, train_preds_bin)
        train_precision = precision_score(train_labels_bin, train_preds_bin, average='macro', zero_division=0)
        train_recall = recall_score(train_labels_bin, train_preds_bin, average='macro', zero_division=0)
        train_f1 = f1_score(train_labels_bin, train_preds_bin, average='macro', zero_division=0)
        train_iou = iou_score(train_labels_bin, train_preds_bin)
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(train_acc)

        print(f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Acc: {train_acc:.4f} | Precision: {train_precision:.4f} | "
              f"Recall: {train_recall:.4f} | F1: {train_f1:.4f} | IoU: {train_iou:.4f}")

        # ----- Validation -----
        model.eval()
        val_preds, val_labels = [], []
        total_val_loss = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                imgs = imgs.to(DEVICE)
                outputs = model(imgs).cpu().numpy().squeeze()
                val_preds.extend(outputs)
                val_labels.extend(labels.numpy())
                total_val_loss += loss.item()

        val_preds_bin = np.round(val_preds)
        val_labels_bin = np.round(val_labels)

        val_acc = accuracy_score(val_labels_bin, val_preds_bin)
        val_losses.append(total_val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        kappa = quadratic_weighted_kappa(val_preds, val_labels)
        cohen = cohen_kappa_score(val_labels_bin, val_preds_bin)
        val_precision = precision_score(val_labels_bin, val_preds_bin, average='macro', zero_division=0)
        val_recall = recall_score(val_labels_bin, val_preds_bin, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels_bin, val_preds_bin, average='macro', zero_division=0)
        val_iou = iou_score(val_labels_bin, val_preds_bin)
        try:
            auc = roc_auc_score(val_labels_bin, val_preds, multi_class='ovr')
        except:
            auc = float('nan')

        duration = time.time() - start_time
        print(f"Validation - Acc: {val_acc:.4f} | QWK: {kappa:.4f} | Kappa: {cohen:.4f} | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | "
              f"F1: {val_f1:.4f} | IoU: {val_iou:.4f} | AUC: {auc:.4f}")
        print(f"⏱️ Epoch Time: {duration:.2f} sec")

        # ----- Save model -----
    model_path = f"dr_model_epoch{epoch+1}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Saved model to {model_path}")

    epochs = np.arange(1, EPOCHS + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Acc")
    plt.plot(epochs, val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Epoch vs Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

    # ----- CONFUSION MATRIX -----
    cm = confusion_matrix(val_labels_bin, val_preds_bin, labels=[0, 1, 2, 3, 4])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.xticks(np.arange(5))
    plt.yticks(np.arange(5))
    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

# ----- ENTRY POINT -----
if __name__ == "__main__":
    train()
