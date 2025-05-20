from tqdm import tqdm
import time
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ----- CONFIG -----
CSV_PATH = './data/messidor_data.csv'
IMG_FOLDER = '../../../Downloads/messidor-2/messidor-2/preprocess/'
BATCH_SIZE = 8
IMAGE_SIZE = 512  # Optimal size for EfficientNet
EPOCHS = 25
LR = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './dr_model_final.pth'

# ----- IMAGE PREPROCESSING -----
def preprocess_image(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return Image.fromarray(img)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

# ----- DATASET -----
class MessidorDataset(Dataset):
    def __init__(self, csv_path, img_folder, transform=None, train=True):
        self.df = pd.read_csv(csv_path)
        self.img_folder = img_folder
        self.transform = transform
        
        # Filter only gradable images (adjudicated_gradable == 1)
        self.df = self.df[self.df['adjudicated_gradable'] == 1]
        
        # Split data (80% train, 20% val)
        train_df, val_df = train_test_split(
            self.df, 
            test_size=0.2, 
            random_state=42,
            stratify=self.df['level']  # Maintain class distribution
        )
        self.df = train_df if train else val_df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.df.iloc[idx]['image'])
        label = self.df.iloc[idx]['level']  # Using 'level' as the target
        
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = preprocess_image(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.float)

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

# ----- METRICS -----
def quadratic_weighted_kappa(y_pred, y_true):
    y_pred = np.clip(np.round(y_pred), 0, 4).astype(int)
    y_true = np.array(y_true).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# ----- TRAINING LOOP -----
def train():
    # Create datasets
    train_ds = MessidorDataset(CSV_PATH, IMG_FOLDER, transform=transform, train=True)
    val_ds = MessidorDataset(CSV_PATH, IMG_FOLDER, transform=transform, train=False)

    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True  # Maintains workers between epochs
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = EnhancedDRModel().to(DEVICE)

        # Load pre-trained model weights if available
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # If it was saved as a dict (e.g. {'state_dict': ..., 'image_size': ..., etc.})
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if trained with DataParallel or DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')  # Handle multi-GPU case
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Loaded pretrained model")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    
    best_qwk = -1
    early_stop_counter = 0
    early_stop_patience = 5

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        start_time = time.time()

        # Training phase
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for imgs, labels in pbar:
                imgs, labels = imgs.to(DEVICE), labels.unsqueeze(1).to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs).cpu().numpy().squeeze()
                val_preds.extend(outputs)
                val_labels.extend(labels.numpy())

        # Calculate metrics
        val_labels_int = np.array(val_labels).astype(int)
        preds_rounded = np.clip(np.round(val_preds), 0, 4).astype(int)
        
        acc = accuracy_score(val_labels_int, preds_rounded)
        kappa = cohen_kappa_score(val_labels_int, preds_rounded)
        qwk = quadratic_weighted_kappa(val_preds, val_labels)

        
        print(f"✅ Val Acc: {acc:.4f}, Kappa: {kappa:.4f}, QWK: {qwk:.4f}")

        # Save best model
        if qwk > best_qwk:
            torch.save(model.state_dict(), './best_model_messidor.pth')
            best_qwk = qwk
            early_stop_counter = 0
            print(f"💾 Saved new best model with QWK: {qwk:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

        # Update learning rate
        scheduler.step(qwk)

if __name__ == '__main__':
    train()