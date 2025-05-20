import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ----- CONFIG -----
BATCH_SIZE = 2
IMAGE_SIZE = 512
EPOCHS = 25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_DR_CLASSES = 5
NUM_EDEMA_CLASSES = 3
SEG_CLASSES = ['Microaneurysms', 'Haemorrhages', 'Soft_Exudates', 'Hard_Exudates', 'Optic_Disc']

# ----- PATHS -----
# Disease Grading Paths
TRAIN_LABELS_EXCEL = '../../../Downloads/Disease_Grading/Groundtruths/IDRiD_Disease Grading_Training Labels.csv'
TEST_LABELS_EXCEL = '../../../Downloads/Disease_Grading/Groundtruths/IDRiD_Disease Grading_Testing Labels.csv'
DISEASE_TRAIN_IMAGES = '../../../Downloads/Disease_Grading/Original_Images/Training_Set'
DISEASE_TEST_IMAGES = '../../../Downloads/Disease_Grading/Original_Images/Testing_Set'

# Segmentation Paths
SEG_TRAIN_IMAGES = '../../../Downloads/Segmentation/Original_Images/Training_Set'
SEG_TEST_IMAGES = '../../../Downloads/Segmentation/Original_Images/Testing_Set'
SEG_BASE_TRAIN = '../../../Downloads/Segmentation/Groundtruths/Training_Set'
SEG_BASE_TEST = '../../../Downloads/Segmentation/Groundtruths/Testing_Set'

# ----- UTILS -----
def read_binary_mask(image_path):
    """Debug version to verify mask loading"""
    if not os.path.exists(image_path):
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    
    mask = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)  # Load with original depth
    if mask is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    
    
    # Normalize based on actual values
    if mask.max() > 0:
        mask = (mask > 0).astype(np.uint8)
    else:
        print(f"WARNING: Empty mask {image_path}")
    
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return mask

# ----- DATASETS -----
class DiseaseGradingDataset(Dataset):
    """Dataset for disease grading task (classification only)"""
    def __init__(self, df, img_folder, transform=None):
        self.df = df
        self.img_folder = img_folder
        self.transform = transform
        self.image_names = sorted([f[:-4] for f in os.listdir(img_folder) if f.endswith('.jpg')])
        self.df = self.df[self.df['Image name'].isin(self.image_names)]
        print(f"Disease Grading Dataset size: {len(self.df)} images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image name']
        dr_label = self.df.iloc[idx]['Retinopathy grade']
        edema_label = self.df.iloc[idx]['Risk of macular edema']

        img_path = os.path.join(self.img_folder, img_name + '.jpg')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(image)

        return image, torch.tensor(dr_label, dtype=torch.long), torch.tensor(edema_label, dtype=torch.long)

class SegmentationDataset(Dataset):
    """Dataset for segmentation task only"""
    def __init__(self, img_folder, seg_folder, transform=None):
        self.img_folder = img_folder
        self.seg_folder = seg_folder
        self.transform = transform
        self.image_names = sorted([f[:-4] for f in os.listdir(img_folder) if f.endswith('.jpg')])
        print(f"Segmentation Dataset size: {len(self.image_names)} images")
        
        # Suffix mapping for segmentation masks
        self.seg_suffix_map = {
            'Microaneurysms': '_MA',
            'Haemorrhages': '_HE',
            'Soft_Exudates': '_SE',
            'Hard_Exudates': '_EX',
            'Optic_Disc': '_OD'
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_folder, img_name + '.jpg')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(image)

        # Load segmentation masks
        mask = np.zeros((len(SEG_CLASSES), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for i, lesion in enumerate(SEG_CLASSES):
            suffix = self.seg_suffix_map[lesion]
            lesion_filename = img_name + suffix + '.tif'
            lesion_path = os.path.join(self.seg_folder, lesion, lesion_filename)
            lesion_mask = read_binary_mask(lesion_path)
            mask[i] = lesion_mask

        return image, torch.tensor(mask, dtype=torch.float)

# ----- MODELS -----
class DiseaseGradingModel(nn.Module):
    """Model for disease grading task only"""
    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Remove the original classifier but keep the features
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
        # Get the number of features from the last layer
        in_features = self.backbone.classifier[1].in_features
        
        # Classification heads
        self.fc_dr = nn.Linear(in_features, NUM_DR_CLASSES)
        self.fc_edema = nn.Linear(in_features, NUM_EDEMA_CLASSES)

    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        
        # Classification outputs
        dr_out = self.fc_dr(x)
        edema_out = self.fc_edema(x)
        return dr_out, edema_out
class SegmentationModel(nn.Module):
    """Model for segmentation task only"""
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).features
        
        # Segmentation decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(1536, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, len(SEG_CLASSES), kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        seg_out = self.decoder(features)
        seg_out = F.interpolate(seg_out, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        return seg_out

# ----- LOSS FUNCTIONS -----
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    numerator = 2 * (pred * target).sum(dim=(2, 3))
    denominator = (pred + target).sum(dim=(2, 3)) + eps
    return 1 - (numerator / denominator).mean()

def combined_seg_loss(pred, target, eps=1e-6):
    dice = dice_loss(pred, target, eps)
    weights = torch.ones_like(target) * 0.1  # Lower weight for background
    weights[target > 0] = 1.0  # Higher weight for lesions
    bce = F.binary_cross_entropy_with_logits(pred, target, weight=weights)
    return 0.5 * dice + 0.5 * bce

# ----- TRAINING FUNCTIONS -----
def train_disease_model(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for img, dr_label, edema_label in tqdm(loader, desc="Training"):
        img, dr_label, edema_label = img.to(device), dr_label.to(device), edema_label.to(device)
        
        dr_pred, edema_pred = model(img)
        loss = criterion(dr_pred, dr_label) + criterion(edema_pred, edema_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Training Loss: {total_loss / len(loader):.4f}")

def train_seg_model(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for img, mask in tqdm(loader, desc="Training"):
        img, mask = img.to(device), mask.to(device)
        
        seg_pred = model(img)
        loss = combined_seg_loss(seg_pred, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Training Loss: {total_loss / len(loader):.4f}")

# ----- EVALUATION FUNCTIONS -----
def evaluate_disease_model(model, loader, criterion, device):
    model.eval()
    all_dr_preds, all_edema_preds = [], []
    all_dr_labels, all_edema_labels = [], []
    total_loss = 0
    
    with torch.no_grad():
        for img, dr_label, edema_label in tqdm(loader, desc="Validation"):
            img, dr_label, edema_label = img.to(device), dr_label.to(device), edema_label.to(device)
            
            dr_pred, edema_pred = model(img)
            loss = criterion(dr_pred, dr_label) + criterion(edema_pred, edema_label)
            total_loss += loss.item()
            
            all_dr_preds.extend(dr_pred.argmax(1).cpu().numpy())
            all_edema_preds.extend(edema_pred.argmax(1).cpu().numpy())
            all_dr_labels.extend(dr_label.cpu().numpy())
            all_edema_labels.extend(edema_label.cpu().numpy())
    
    acc_dr = accuracy_score(all_dr_labels, all_dr_preds)
    acc_edema = accuracy_score(all_edema_labels, all_edema_preds)
    kappa = cohen_kappa_score(all_dr_labels, all_dr_preds, weights='quadratic')
    
    print(f"Validation Loss: {total_loss / len(loader):.4f}")
    print(f"DR Accuracy: {acc_dr:.4f} | Edema Accuracy: {acc_edema:.4f} | Kappa: {kappa:.4f}")

def iou_score(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).bool()
    target = target.bool()
    intersection = (pred & target).float().sum((2, 3))
    union = (pred | target).float().sum((2, 3))
    # Handle cases where union is zero
    iou = torch.where(union > 0, (intersection + eps) / (union + eps), torch.tensor(0.0, device=pred.device))
    return iou.mean().item()

def evaluate_seg_model(model, loader, device):
    model.eval()
    iou_total = 0
    num_batches = 0
    
    with torch.no_grad():
        for img, mask in tqdm(loader, desc="Validation"):
            img, mask = img.to(device), mask.to(device)
            
            seg_pred = model(img)
            print("Pred min:", seg_pred.min().item(), "max:", seg_pred.max().item())
            iou_total += iou_score(seg_pred, mask)
            num_batches += 1
    
    mean_iou = iou_total / num_batches if num_batches > 0 else 0
    print(f"Mean IoU: {mean_iou:.4f}")

# ----- MAIN -----
if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv(TRAIN_LABELS_EXCEL)
    test_df = pd.read_csv(TEST_LABELS_EXCEL)
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    disease_train_dataset = DiseaseGradingDataset(train_df, DISEASE_TRAIN_IMAGES, transform)
    disease_test_dataset = DiseaseGradingDataset(test_df, DISEASE_TEST_IMAGES, transform)
    
    seg_train_dataset = SegmentationDataset(SEG_TRAIN_IMAGES, SEG_BASE_TRAIN, transform)
    seg_test_dataset = SegmentationDataset(SEG_TEST_IMAGES, SEG_BASE_TEST, transform)

    # Create dataloaders
    disease_train_loader = DataLoader(disease_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    disease_test_loader = DataLoader(disease_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    seg_train_loader = DataLoader(seg_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    seg_test_loader = DataLoader(seg_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nTraining Segmentation Model...")
    seg_model = SegmentationModel().to(DEVICE)
    seg_optimizer = torch.optim.Adam(seg_model.parameters(), lr=1e-4)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_seg_model(seg_model, seg_train_loader, seg_optimizer, DEVICE)
        evaluate_seg_model(seg_model, seg_test_loader, DEVICE)
        torch.save(seg_model.state_dict(), f'seg_model_epoch_{epoch+1}.pth')

    # Train disease grading model
    print("\nTraining Disease Grading Model...")
    disease_model = DiseaseGradingModel().to(DEVICE)
    disease_optimizer = torch.optim.Adam(disease_model.parameters(), lr=1e-4)
    disease_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_disease_model(disease_model, disease_train_loader, disease_optimizer, disease_criterion, DEVICE)
        evaluate_disease_model(disease_model, disease_test_loader, disease_criterion, DEVICE)
        torch.save(disease_model.state_dict(), f'disease_model_epoch_{epoch+1}.pth')

    # Train segmentation model
    print("\nTraining Segmentation Model...")
    seg_model = SegmentationModel().to(DEVICE)
    seg_optimizer = torch.optim.Adam(seg_model.parameters(), lr=1e-4)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_seg_model(seg_model, seg_train_loader, seg_optimizer, DEVICE)
        evaluate_seg_model(seg_model, seg_test_loader, DEVICE)
        torch.save(seg_model.state_dict(), f'seg_model_epoch_{epoch+1}.pth')