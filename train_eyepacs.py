from sklearn.metrics import accuracy_score, f1_score

from multiprocessing import Pool, cpu_count
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
# eyepacs_effnetb3_pipeline.py

import os
import h5py
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from torchvision import transforms
from tqdm import tqdm
from albumentations import (Compose, CLAHE, RandomRotate90, HorizontalFlip, VerticalFlip, Normalize, Resize, GaussNoise, RandomBrightnessContrast)
from albumentations.pytorch import ToTensorV2
import timm

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
RAW_H5 = './data/eyepacs.h5'
PREPROCESSED_H5 = 'eyepacs_preprocessed.h5'

def process_single_image(img):
    IMAGE_SIZE = 256
    img = crop_black_border(img)
    img = apply_clahe(img)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def preprocess_split(images):
    with Pool(cpu_count()) as pool:  # Use all CPU cores
        processed = list(tqdm(pool.imap(process_single_image, images), total=len(images)))
    return processed

def preprocess_and_save():
    if os.path.exists(PREPROCESSED_H5):
        print("Preprocessed file already exists. Skipping preprocessing.")
        return

    print("Starting preprocessing...")
    if not os.path.exists(RAW_H5):
        raise FileNotFoundError(f"{RAW_H5} not found. Please ensure the raw HDF5 file exists.")

    IMAGE_SIZE = 256

    with h5py.File(RAW_H5, 'r') as f:
        # First get all data into memory (if it fits)
        train_images = f['train/images'][:]
        train_labels = f['train/labels'][:]
        val_images = f['val/images'][:]
        val_labels = f['val/labels'][:]

    # Process in parallel
    with Pool(cpu_count()) as pool:
        print("Processing training images...")
        processed_train = list(tqdm(pool.imap(process_single_image, train_images), total=len(train_images)))
        
        print("Processing validation images...")
        processed_val = list(tqdm(pool.imap(process_single_image, val_images), total=len(val_images)))

    # Save processed data
    with h5py.File(PREPROCESSED_H5, 'w') as out_f:
        out_f.create_dataset('train/images', data=np.array(processed_train), compression='gzip')
        out_f.create_dataset('train/labels', data=train_labels)
        out_f.create_dataset('val/images', data=np.array(processed_val), compression='gzip')
        out_f.create_dataset('val/labels', data=val_labels)

    print("Preprocessing finished")

def crop_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img  # fallback if no contour is found

    x, y, w, h = cv2.boundingRect(contours[0])
    return img[y:y+h, x:x+w]


def combined_loss(pred, target, alpha=0.7):
    ce = nn.CrossEntropyLoss()(pred, target)
    smooth = smooth_mse_loss(torch.argmax(pred, dim=1), target)
    return alpha * ce + (1 - alpha) * smooth



def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

# ----------------------
# Dataset & Augmentations
# ----------------------
class EyePACSDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = img.astype(np.float32) / 255.0

        if self.transform:
            img = self.transform(image=img)['image']
        return img, torch.tensor(label, dtype=torch.long)



def get_transforms(train=True):
    if train:
        return Compose([
            Resize(256, 256),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            RandomBrightnessContrast(p=0.3),
            GaussNoise(p=0.3),
            Normalize(),
            ToTensorV2()
        ])
    else:
        return Compose([
            Resize(256, 256),
            Normalize(),
            ToTensorV2()
        ])


# ----------------------
# Model Definition
# ----------------------
class DRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(in_features // 2, 5)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

# ----------------------
# Loss Functions
# ----------------------
def smooth_mse_loss(pred, target, smoothing=0.1):
    pred = pred.unsqueeze(1) if pred.ndim == 1 else pred  # shape [B, 1]
    pred = torch.clamp(pred.repeat(1, 5), 0, 4)           # shape [B, 5]

    target = target.long()
    one_hot = torch.zeros((target.size(0), 5), device=target.device)
    one_hot.scatter_(1, target.unsqueeze(1), 1)
    one_hot = one_hot * (1 - smoothing) + (smoothing / 5)

    return nn.MSELoss()(pred, one_hot)


def qwk_metric(y_true, y_pred):
    y_pred = np.clip(np.round(y_pred), 0, 4).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# ----------------------
# Training Loop
# ----------------------
def train_kfold(images, labels, n_splits=5, num_epochs=15, batch_size=16):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\nFOLD {fold+1}")
        X_train, X_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_dataset = EyePACSDataset(X_train, y_train, transform=get_transforms(train=True))
        val_dataset = EyePACSDataset(X_val, y_val, transform=get_transforms(train=False))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = DRModel().to(device)
        model.freeze_backbone()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        # criterion = nn.CrossEntropyLoss()   

        best_kappa = -np.inf
        patience, wait = 5, 0

        for epoch in range(num_epochs):
            model.train()
            train_losses = []

            # Unfreeze backbone after a few epochs
            if epoch == 5:
                print("Unfreezing backbone for fine-tuning...")
                model.unfreeze_backbone()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - epoch)

            for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)  # <-- ADDED THIS LINE
                loss = combined_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()

            # Validation phase
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(targets.numpy())

            # Evaluation metrics
            acc = accuracy_score(all_targets, all_preds)
            f1 = f1_score(all_targets, all_preds, average='weighted')
            kappa = qwk_metric(np.array(all_targets), np.array(all_preds))

            print(f"Epoch {epoch+1} -> Loss: {np.mean(train_losses):.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | QWK: {kappa:.4f}")

            # Save best model
            if kappa > best_kappa:
                best_kappa = kappa
                wait = 0
                torch.save(model.state_dict(), f"best_model_fold{fold}.pt")
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping.")
                    break


# ----------------------
# TTA Inference Function
# ----------------------
def tta_predict(model, img):
    model.eval()
    tfs = [
        get_transforms(train=False),
        Compose([HorizontalFlip(p=1.0), Normalize(), ToTensorV2()]),
        Compose([VerticalFlip(p=1.0), Normalize(), ToTensorV2()]),
        Compose([RandomRotate90(p=1.0), Normalize(), ToTensorV2()])
    ]
    preds = []
    for tfm in tfs:
        img_tensor = tfm(image=img)['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor)
            pred_class = torch.softmax(pred, dim=1)
            preds.append(pred_class.cpu().numpy())
    return np.mean(preds, axis=0).argmax()


# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    preprocess_and_save()

    with h5py.File(PREPROCESSED_H5, 'r') as f:
        train_images = f['train/images'][:]
        train_labels = f['train/labels'][:]
        val_images = f['val/images'][:]
        val_labels = f['val/labels'][:]

    train_kfold(train_images, train_labels)

    # Load the best model from a fold (e.g., fold0) to run on validation set
    model = DRModel().to(device)
    model.load_state_dict(torch.load("best_model_fold0.pt", map_location=device))

    print("\nRunning TTA on validation set...")
    predictions = []
    for img in tqdm(val_images):
        pred = tta_predict(model, img)
        predictions.append(pred)

    kappa = qwk_metric(val_labels, predictions)
    acc = accuracy_score(val_labels, predictions)
    f1 = f1_score(val_labels, predictions, average='weighted')
    print(f"\nTTA Results -> Acc: {acc:.4f} | F1: {f1:.4f} | QWK: {kappa:.4f}")
