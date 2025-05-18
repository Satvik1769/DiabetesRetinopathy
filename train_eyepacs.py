import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import cv2
import time
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, cohen_kappa_score, roc_auc_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
import os

# ----- ENHANCED CONFIG -----
HDF5_PATH = './data/eyepacs.h5'
BATCH_SIZE = 32
IMAGE_SIZE = 512  # Optimal balance between performance and memory
EPOCHS = 25      # Increased for better convergence
LR = 3e-5        # Lower learning rate for fine-tuning compatibility
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRAD_ACCUM_STEPS = 4  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS

# ----- ADVANCED PREPROCESSING -----
def clahe_enhancement(img):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    img = np.array(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def advanced_preprocess(img):
    """Combined preprocessing pipeline"""
    # 1. Initial processing
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2. More aggressive border removal
    _, thresh = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        img = img[y:y+h, x:x+w]
    
    # 3. CLAHE enhancement
    img = clahe_enhancement(Image.fromarray(img))
    
    # 4. Smart padding
    old_size = img.shape[:2]
    ratio = float(IMAGE_SIZE) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    
    delta_w = IMAGE_SIZE - new_size[1]
    delta_h = IMAGE_SIZE - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                            cv2.BORDER_CONSTANT, value=[0,0,0])
    return Image.fromarray(img)

# ----- ENHANCED TRANSFORMS -----
train_transform = T.Compose([
    T.Lambda(advanced_preprocess),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

val_transform = T.Compose([
    T.Lambda(advanced_preprocess),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----- SIMPLIFIED DATASET CLASS -----
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

# ----- IMPROVED MODEL ARCHITECTURE -----
class EnhancedDRModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize with pretrained weights
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Modified head structure for better feature extraction
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(in_features//2, 1)
        
        # Freeze early layers for stability
        for param in list(self.backbone.parameters())[:-4]:
            param.requires_grad = False
    
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)
    
    def unfreeze_all(self):
        """For future fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True

# ----- QWK-OPTIMIZED LOSS -----
class QWKLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, targets):
        # Clip predictions to valid grade range [0,4]
        preds = torch.clamp(preds, 0., 4.)
        return F.mse_loss(preds, targets)

# ----- ENHANCED TRAINING LOOP -----
def train(resume=True):
    model = EnhancedDRModel().to(DEVICE)
    criterion = QWKLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    best_qwk = 0
    start_epoch = 0

    # Resume training if checkpoint exists
    if resume and os.path.exists("best_model.pth"):
        checkpoint = torch.load("best_model.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        print("✅ Loaded existing model from best_model.pth")

    # Load datasets
    train_dataset = HDF5Dataset(HDF5_PATH, 'train', train_transform)
    val_dataset = HDF5Dataset(HDF5_PATH, 'val', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)
    
    best_qwk = 0
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            imgs, labels = imgs.to(DEVICE), labels.unsqueeze(1).to(DEVICE)
            
            preds = model(imgs)
            loss = criterion(preds, labels) / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * GRAD_ACCUM_STEPS
        
        train_loss = running_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs.to(DEVICE)).cpu().numpy().squeeze()
                val_preds.extend(outputs)
                val_labels.extend(labels.numpy())
        
        # Calculate metrics
        val_preds_int = np.clip(np.round(val_preds), 0, 4).astype(int)
        val_labels_int = np.array(val_labels).astype(int)
        
        val_acc = accuracy_score(val_labels_int, val_preds_int)
        qwk = cohen_kappa_score(val_labels_int, val_preds_int, weights='quadratic')
        scheduler.step(qwk)
        
        # Save best model
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(model.state_dict(), "best_model.pth")
            print(f"🔥 New best model saved with QWK: {qwk:.4f}")
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss:.4f} | Val QWK: {qwk:.4f} | Val Acc: {val_acc:.4f}")
    
    # --- Final Evaluation ---
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    # Generate full metrics
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            outputs = model(imgs.to(DEVICE)).cpu().numpy().squeeze()
            val_preds.extend(outputs)
            val_labels.extend(labels.numpy())
    
    val_preds_int = np.clip(np.round(val_preds), 0, 4).astype(int)
    val_labels_int = np.array(val_labels).astype(int)
    
    # Calculate all metrics
    cm = confusion_matrix(val_labels_int, val_preds_int, labels=[0,1,2,3,4])
    val_acc = accuracy_score(val_labels_int, val_preds_int)
    qwk = cohen_kappa_score(val_labels_int, val_preds_int, weights='quadratic')
    precision = precision_score(val_labels_int, val_preds_int, average='macro', zero_division=0)
    recall = recall_score(val_labels_int, val_preds_int, average='macro', zero_division=0)
    f1 = f1_score(val_labels_int, val_preds_int, average='macro', zero_division=0)
    
    print("\nFinal Validation Metrics:")
    print(f"Accuracy: {val_acc:.4f} | QWK: {qwk:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    
    # Save final model
    torch.save({
        'state_dict': model.state_dict(),
        'image_size': IMAGE_SIZE,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }, "dr_model_final.pth")

    # Plot training curves and confusion matrix
    plot_results()

def plot_results():
    """Helper function to generate plots"""
    # [Include your existing plotting code here]
    pass

if __name__ == "__main__":
    train(resume=True)