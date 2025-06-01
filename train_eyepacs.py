import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.serialization as torch_serialization
import warnings
warnings.filterwarnings('ignore')

# Allowlist NumPy global to fix WeightsUnpickler error
torch_serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

# ----- IMPROVED CONFIG -----
HDF5_PATH = './data/eyepacs_preprocessed.h5'
BATCH_SIZE = 12  # Slightly reduced for stability
IMAGE_SIZE = 448
EPOCHS = 30
LR = 2e-4  # Increased learning rate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRAD_ACCUM_STEPS = 2  # Reduced for faster updates
WARMUP_EPOCHS = 5

# FIXED: Increase NUM_WORKERS even on Windows
NUM_WORKERS = 0 if torch.cuda.is_available() else 4  # Use 8 workers for GPU training

CHECKPOINT_PATH = "best_enhanced_model.pth"

# ----- SIMPLIFIED PREPROCESSING FOR PERFORMANCE -----
def enhanced_fundus_preprocess(img):
    """Optimized preprocessing with reduced computational overhead"""
    img = np.array(img)  
    
    # Simplified contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Faster CLAHE with smaller tile size
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
    l = clahe.apply(l)
    
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    
    # Simplified cropping - just center crop for speed
    h, w = img.shape[:2]
    size = min(h, w)
    x1, y1 = (w - size) // 2, (h - size) // 2
    img = img[y1:y1+size, x1:x1+size]
    
    # Faster resize method
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    
    return Image.fromarray(img)

def preprocess_wrapper(image, **kwargs):
    return np.array(enhanced_fundus_preprocess(Image.fromarray(image)))

# ----- OPTIMIZED AUGMENTATIONS -----
train_transform = A.Compose([
    A.Lambda(image=preprocess_wrapper),
    # Reduced augmentation complexity for speed
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(
        scale=(0.9, 1.1), 
        translate_percent=0.1, 
        rotate=(-15, 15), 
        p=0.5
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2, 
        contrast_limit=0.2, 
        p=0.6
    ),
    A.HueSaturationValue(
        hue_shift_limit=10, 
        sat_shift_limit=15, 
        val_shift_limit=10, 
        p=0.4
    ),
    # Reduced noise augmentation
    A.OneOf([
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.5),
    ], p=0.2),
    # Final normalization
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Lambda(image=preprocess_wrapper),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ----- OPTIMIZED DATASET WITH CACHING -----
class ImprovedHDF5Dataset(Dataset):
    def __init__(self, file_path, split, transform=None, oversample_factor=2.0, cache_data=True):
        self.file_path = file_path
        self.split = split
        self.transform = transform
        self.cache_data = cache_data
        self.cached_images = {}
        
        with h5py.File(self.file_path, 'r') as f:
            image_key = f'{split}/images'
            label_key = f'{split}/labels'
            if image_key not in f or label_key not in f:
                raise KeyError(f"Expected keys '{image_key}' and '{label_key}' not found")
            
            self.images = f[image_key][:]
            self.labels = f[label_key][:]
        
        # Create balanced dataset through strategic oversampling
        self.class_counts = np.bincount(self.labels, minlength=5)
        max_count = self.class_counts.max()
        
        # Create oversampled indices
        self.indices = []
        for class_idx in range(5):
            class_indices = np.where(self.labels == class_idx)[0]
            if len(class_indices) == 0:
                continue
            
            # Calculate how many times to repeat each class
            target_count = int(max_count * oversample_factor) if class_idx > 0 else max_count
            repeats = max(1, target_count // len(class_indices))
            
            # Add indices with repetition
            for _ in range(repeats):
                self.indices.extend(class_indices)
            
            # Add remaining samples randomly
            remaining = target_count - (repeats * len(class_indices))
            if remaining > 0:
                self.indices.extend(np.random.choice(class_indices, remaining, replace=True))
        
        self.indices = np.array(self.indices)
        np.random.shuffle(self.indices)
        
        print(f"Dataset {split}: Original {len(self.labels)}, Oversampled {len(self.indices)}")
        print(f"Class distribution after oversampling: {Counter(self.labels[self.indices])}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        # Use cached image if available
        if self.cache_data and actual_idx in self.cached_images:
            image = self.cached_images[actual_idx]
        else:
            image = self.images[actual_idx]
            if self.cache_data and len(self.cached_images) < 1000:  # Cache first 1000 images
                self.cached_images[actual_idx] = image.copy()
        
        label = self.labels[actual_idx]
        
        if self.transform:
            try:
                transformed = self.transform(image=image)
                image = transformed['image']
            except Exception as e:
                # Fallback to basic transform
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
        return image, torch.tensor(label, dtype=torch.long)  # Ensure long tensor

# ----- IMPROVED MODEL ARCHITECTURE -----
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ImprovedDRModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Load pretrained EfficientNet-B4
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        
        # Get the number of features before classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Add attention module
        self.attention = CBAM(1792)  # EfficientNet-B4 has 1792 features
        
        # Improved classifier with multiple heads
        self.feature_extractor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Regression head for ordinal regression
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Gradual unfreezing strategy
        self.unfreeze_schedule = {
            0: ['classifier', 'regressor', 'feature_extractor', 'attention'],
            10: ['features.7', 'features.6'],  # Last two blocks
            20: ['features.5', 'features.4'],  # Middle blocks
            30: ['features.3', 'features.2'],  # Earlier blocks
        }
        
        self.current_epoch = 0
        self._update_frozen_layers()
    
    def _initialize_weights(self):
        for m in [self.feature_extractor, self.classifier, self.regressor]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
    
    def _update_frozen_layers(self):
        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze based on current epoch
        for epoch_threshold, layers_to_unfreeze in self.unfreeze_schedule.items():
            if self.current_epoch >= epoch_threshold:
                for layer_name in layers_to_unfreeze:
                    for name, param in self.backbone.named_parameters():
                        if layer_name in name:
                            param.requires_grad = True
    
    def update_epoch(self, epoch):
        old_epoch = self.current_epoch
        self.current_epoch = epoch
        if old_epoch != epoch:
            self._update_frozen_layers()
    
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Global average pooling
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Feature extraction
        features = self.feature_extractor(features)
        
        # Classification and regression outputs
        classification_out = self.classifier(features)
        regression_out = self.regressor(features)
        
        return classification_out, regression_out

# ----- FIXED LOSS FUNCTION -----
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        # FIXED: Ensure targets are long tensor
        targets = targets.long()
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, predictions, targets):
        """
        Ordinal regression loss that preserves order relationships
        """
        targets = targets.long()
        batch_size = predictions.size(0)
        
        # Create ordinal targets (cumulative labels)
        ordinal_targets = torch.zeros(batch_size, self.num_classes - 1).to(predictions.device)
        for i in range(self.num_classes - 1):
            ordinal_targets[:, i] = (targets > i).float()
        
        # Apply sigmoid to predictions for ordinal regression
        predictions = torch.sigmoid(predictions)
        
        # Binary cross entropy for each ordinal class
        loss = F.binary_cross_entropy(predictions, ordinal_targets)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, num_classes=5):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=1, gamma=2, class_weights=class_weights)
        self.mse_loss = nn.MSELoss()
        self.ordinal_loss = OrdinalRegressionLoss(num_classes)
        
    def forward(self, cls_preds, reg_preds, targets):
        # FIXED: Ensure targets are long tensor
        targets = targets.long()
        
        # Classification loss (Focal Loss for imbalanced data)
        cls_loss = self.focal_loss(cls_preds, targets)
        
        # Regression loss
        reg_loss = self.mse_loss(reg_preds.squeeze(), targets.float())
        
        # Combined loss with adaptive weighting
        total_loss = 0.7 * cls_loss + 0.3 * reg_loss
        
        return total_loss, cls_loss, reg_loss

# ----- IMPROVED TRAINING FUNCTION -----
def train_improved():
    # Load datasets with oversampling
    try:
        train_dataset = ImprovedHDF5Dataset(HDF5_PATH, 'train', train_transform, oversample_factor=1.5, cache_data=True)
        val_dataset = ImprovedHDF5Dataset(HDF5_PATH, 'val', val_transform, oversample_factor=1.0, cache_data=True)
    except KeyError as e:
        print(f"Error loading dataset: {e}")
        return
    
    # FIXED: Optimized data loaders with performance improvements
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,  # Now uses 8 workers
        pin_memory=True,  # Faster GPU transfer
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS // 2,  # Use fewer workers for validation
        pin_memory=True,
    )
    
    # Initialize model
    model = ImprovedDRModel().to(DEVICE)
    
    # Calculate class weights for loss function
    unique_labels = train_dataset.labels[train_dataset.indices]
    class_weights = compute_class_weight('balanced', classes=np.unique(unique_labels), y=unique_labels)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    # Loss function and optimizer
    criterion = CombinedLoss(class_weights)
    
    # Different learning rates for different parts
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR * 0.1, 'weight_decay': 1e-4},
        {'params': head_params, 'lr': LR, 'weight_decay': 1e-3}
    ])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Training tracking
    best_qwk = 0
    patience_counter = 0
    patience = 15
    
    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            best_qwk = checkpoint['qwk']
            print(f"✅ Resumed from epoch {start_epoch} with QWK: {best_qwk:.4f}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    print(f"Starting training from epoch {start_epoch}")
    train_losses = []
    train_cls_losses = []
    train_reg_losses = []
    train_accuracies = []
    train_qwks = []

    val_accuracies = []
    val_qwks = []
    val_f1s = []

    
    for epoch in range(start_epoch, EPOCHS):
        # Update model's epoch for gradual unfreezing
        model.update_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_reg_loss = 0.0
        
        train_preds = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            # FIXED: Use non_blocking for faster GPU transfer
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                cls_outputs, reg_outputs = model(images)
                total_loss, cls_loss, reg_loss = criterion(cls_outputs, reg_outputs, labels)
                total_loss = total_loss / GRAD_ACCUM_STEPS
            
            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            
            # Track metrics
            train_loss += total_loss.item() * GRAD_ACCUM_STEPS
            train_cls_loss += cls_loss.item()
            train_reg_loss += reg_loss.item()
            
            # Store predictions for epoch-level metrics
            with torch.no_grad():
                preds = torch.argmax(cls_outputs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item() * GRAD_ACCUM_STEPS:.4f}',
                'Cls': f'{cls_loss.item():.4f}',
                'Reg': f'{reg_loss.item():.4f}'
            })
        
        # Calculate training metrics
        train_acc = accuracy_score(train_targets, train_preds)
        train_qwk = cohen_kappa_score(train_targets, train_preds, weights='quadratic')
        
        # Validation phase
        val_qwk, val_acc, val_f1 = evaluate_improved(model, val_loader)
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{EPOCHS} Results:")
        print(f"Train - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}, QWK: {train_qwk:.4f}")
        print(f"Val - Acc: {val_acc:.4f}, QWK: {val_qwk:.4f}, F1: {val_f1:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        # Store metrics for plotting
        train_losses.append(train_loss / len(train_loader))
        train_cls_losses.append(train_cls_loss / len(train_loader))
        train_reg_losses.append(train_reg_loss / len(train_loader))
        train_accuracies.append(train_acc)
        train_qwks.append(train_qwk)

        val_accuracies.append(val_acc)
        val_qwks.append(val_qwk)
        val_f1s.append(val_f1)

        
        # Save best model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            patience_counter = 0
            print(f"💾 New best model! QWK: {best_qwk:.4f}")
            
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'qwk': best_qwk,
                'acc': val_acc
            }, CHECKPOINT_PATH)
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break
        
        # Plot confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_confusion_matrix(model, val_loader, epoch + 1)
    
    print(f"\n🎉 Training completed! Best QWK: {best_qwk:.4f}")
    plot_training_curves(train_losses, train_cls_losses, train_reg_losses,
                     train_accuracies, train_qwks,
                     val_accuracies, val_qwks, val_f1s)
       

def evaluate_improved(model, val_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(DEVICE, non_blocking=True)

            # Original
            cls_out1, reg_out1 = model(images)

            # Horizontal flip TTA
            flipped_images = torch.flip(images, dims=[3])  # Flip along width
            cls_out2, reg_out2 = model(flipped_images)

            # Average predictions
            cls_outputs = (cls_out1 + cls_out2) / 2
            reg_outputs = (reg_out1 + reg_out2) / 2

            # Softmax and regression class conversion
            cls_probs = F.softmax(cls_outputs, dim=1)
            reg_preds = torch.clamp(reg_outputs.squeeze(), 0, 4)
            reg_classes = torch.round(reg_preds).long()
            reg_probs = torch.zeros_like(cls_probs)
            reg_probs.scatter_(1, reg_classes.unsqueeze(1), 1.0)

            # Weighted ensemble
            ensemble_probs = 0.7 * cls_probs + 0.3 * reg_probs
            final_preds = torch.argmax(ensemble_probs, dim=1)

            all_preds.extend(final_preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    return qwk, accuracy, f1

def plot_confusion_matrix(model, val_loader, epoch):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            cls_outputs, _ = model(images)
            preds = torch.argmax(cls_outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                yticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, train_cls_losses, train_reg_losses,
                         train_accuracies, train_qwks,
                         val_accuracies, val_qwks, val_f1s):

    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(20, 12))

    # Total Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Classification Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, train_cls_losses, label='Train Cls Loss')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Regression Loss
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, train_reg_losses, label='Train Reg Loss')
    plt.title('Regression Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # QWK
    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, train_qwks, label='Train QWK')
    plt.plot(epochs_range, val_qwks, label='Val QWK')
    plt.title('Quadratic Weighted Kappa')
    plt.xlabel('Epoch')
    plt.ylabel('QWK')
    plt.legend()

    # F1 Score
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, val_f1s, label='Val F1 Score')
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()

    plt.tight_layout()
    plt.show()


# ----- MAIN EXECUTION -----
if __name__ == "__main__":
    train_improved()
    