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
CSV_PATH = './data/messidor_labels.csv'
IMG_FOLDER = '../../../Downloads/messidor-2/messidor-2/preprocess/'
BATCH_SIZE = 8
IMAGE_SIZE = 728
EPOCHS = 10
LR = 1e-5  # Lower LR for fine-tuning
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './eyepacs.pt'

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
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----- DATASET -----
class PreprocessedDataset(Dataset):
    def __init__(self, image_file, label_file, transform=None):
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx].astype(np.uint8))
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

# ----- METRICS -----
def quadratic_weighted_kappa(y_pred, y_true):
    y_pred = np.clip(np.round(y_pred), 0, 4).astype(int)
    y_true = np.array(y_true).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# ----- TRAINING LOOP -----
def train():

    train_ds = PreprocessedDataset(
        './data/preprocessed/train_images.npy',
        './data/preprocessed/train_labels.npy',
        transform=transform
    )
    val_ds = PreprocessedDataset(
        './data/preprocessed/val_images.npy',
        './data/preprocessed/val_labels.npy',
        transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DRModel().to(DEVICE)

    # Load pre-trained model weights from EyePACS
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Loaded pretrained model from EyePACS.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        num_batches = len(train_loader)
        start_time = time.time()  # Start timing the entire epoch

        # Wrap the train_loader with tqdm for the progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", total=num_batches) as pbar:
            for batch_idx, (imgs, labels) in enumerate(pbar):
                imgs, labels = imgs.to(DEVICE), labels.unsqueeze(1).to(DEVICE)
                preds = model(imgs)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Calculate elapsed time for each batch
                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / (batch_idx + 1)) * (num_batches - (batch_idx + 1))
                remaining_minutes = int(remaining_time // 60)
                remaining_seconds = int(remaining_time % 60)

                # Update tqdm description with remaining time
                pbar.set_postfix(loss=loss.item(), remaining=f"{remaining_minutes}m {remaining_seconds}s")

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs).cpu().numpy().squeeze()
                val_preds.extend(outputs)
                val_labels.extend(labels.numpy())

        preds_rounded = np.clip(np.round(val_preds), 0, 4).astype(int)
        val_labels_int = np.array(val_labels).astype(int)

        acc = accuracy_score(val_labels_int, preds_rounded)
        kappa = cohen_kappa_score(val_labels_int, preds_rounded)
        qwk = quadratic_weighted_kappa(val_preds, val_labels)
       
        try:
            auc = roc_auc_score(val_labels_int, val_preds, multi_class='ovr')
        except:
            auc = -1
        
        print(f"✅ Val Acc: {acc:.4f}, AUC: {auc:.4f}, Kappa: {kappa:.4f}, QWK: {qwk:.4f}")

        # Save model at end of each epoch
        torch.save(model.state_dict(), f'./model_messidor_epoch{epoch+1}.pth')
        print(f"💾 Model saved at ./model_messidor_epoch{epoch+1}.pth")


# ----- ENTRY POINT -----
if __name__ == '__main__':
    train()
