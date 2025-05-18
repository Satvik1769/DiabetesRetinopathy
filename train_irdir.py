import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# ----- CONFIG -----
BATCH_SIZE = 2
IMAGE_SIZE = 512
EPOCHS = 10
LR = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRETRAINED_MODEL_PATH = 'model_messidor.pth'  # Your previous DRModel

NUM_DR_CLASSES = 5
NUM_EDEMA_CLASSES = 3

# ----- PATHS -----
TRAIN_LABELS_EXCEL = '../../../Downloads/Disease_Grading/Groundtruths/IDRiD_Disease Grading_Training Labels.csv'
TEST_LABELS_EXCEL = '../../../Downloads/Disease_Grading/Groundtruths/IDRiD_Disease Grading_Testing Labels.csv'
DISEASE_TRAIN_IMAGES = '../../../Downloads/Disease_Grading/Original_Images/Training_Set'
DISEASE_TEST_IMAGES = '../../../Downloads/Disease_Grading/Original_Images/Testing_Set'

SEG_CLASSES = ['Microaneurysms', 'Haemorrhages', 'Soft_Exudates', 'Hard_Exudates', 'Optic_Disc']
SEG_BASE = '../../../Downloads/Segmentation/Groundtruths/Training_Set'
SEG_BASE_TEST = '../../../Downloads/Segmentation/Groundtruths/Testing_Set'
SEGMENT_TRAIN_IMAGES = '../../../Downloads/Segmentation/Original_Images/Training_Set'
SEGMENT_TEST_IMAGES = '../../../Downloads/Segmentation/Original_Images/Testing_Set'

# ----- UTILS -----
def read_binary_mask(image_path):
    if not os.path.exists(image_path):
        return None
    mask = cv2.imread(image_path)
    if mask is None:
        return None
    red_channel = mask[..., 2]
    lesion = (red_channel > 127).astype(np.uint8)
    return lesion

# ----- DATASETS -----
class ClassificationDataset(Dataset):
    def __init__(self, df, img_folder, transform=None):
        self.df = df
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image name']
        dr_label = row['Retinopathy grade']
        edema_label = row['Risk of macular edema']

        img_path = os.path.join(self.img_folder, img_name + '.jpg')
        image = Image.open(img_path).convert('RGB')
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, torch.tensor(dr_label), torch.tensor(edema_label)
    

SEG_SUFFIX_MAP = {
    'Microaneurysms': '_MA',
    'Haemorrhages': '_HE',
    'Soft_Exudates': '_SE',
    'Hard_Exudates': '_EX',
    'Optic_Disc': '_OD'
}

class SegmentationDataset(Dataset):
    def __init__(self, img_folder, seg_folder, transform=None):
        self.img_folder = img_folder
        self.seg_folder = seg_folder
        self.image_names = sorted([f[:-4] for f in os.listdir(img_folder) if f.endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_folder, img_name + '.jpg')
        image = Image.open(img_path).convert('RGB')
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = np.zeros((5, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for i, lesion in enumerate(SEG_CLASSES):
            suffix = SEG_SUFFIX_MAP[lesion]
            lesion_filename = img_name + suffix + '.tif'
            lesion_path = os.path.join(self.seg_folder, lesion, lesion_filename).replace('\\', '/')
            lesion_mask = read_binary_mask(lesion_path)
            if lesion_mask is not None:
                lesion_mask = cv2.resize(lesion_mask, (IMAGE_SIZE, IMAGE_SIZE))
                mask[i] = lesion_mask / 255.0

        return image, torch.tensor(mask)

# ----- MODEL -----
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b3(weights=None)
        self.backbone.classifier = nn.Identity()
        in_features = 1536  # EfficientNet-B3 output

        self.fc_dr = nn.Linear(in_features, NUM_DR_CLASSES)
        self.fc_edema = nn.Linear(in_features, NUM_EDEMA_CLASSES)

        # Simple upsampling segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_features, 256, kernel_size=1),
            nn.ReLU(),
            nn.Upsample(size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False),
            nn.Conv2d(256, len(SEG_CLASSES), kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)  # (B, 1536)
        dr_out = self.fc_dr(features)
        edema_out = self.fc_edema(features)

        # Simulate spatial feature map for segmentation
        B = x.size(0)
        spatial_feat = features.view(B, -1, 1, 1).expand(B, -1, IMAGE_SIZE, IMAGE_SIZE)
        seg_out = self.segmentation_head(spatial_feat)
        return dr_out, edema_out, seg_out

# ----- LOSS -----
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    numerator = 2 * (pred * target).sum(dim=(2, 3))
    denominator = (pred + target).sum(dim=(2, 3)) + eps
    return 1 - (numerator / denominator).mean()

# ----- LOAD DATA -----
train_df = pd.read_csv(TRAIN_LABELS_EXCEL)
test_df = pd.read_csv(TEST_LABELS_EXCEL)
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

train_cls_dataset = ClassificationDataset(train_df, DISEASE_TRAIN_IMAGES, transform=transform)
test_cls_dataset = ClassificationDataset(test_df, DISEASE_TEST_IMAGES, transform=transform)
train_seg_dataset = SegmentationDataset(SEGMENT_TRAIN_IMAGES, SEG_BASE, transform=transform)
test_seg_dataset = SegmentationDataset(SEGMENT_TEST_IMAGES, SEG_BASE_TEST, transform=transform)

train_cls_loader = DataLoader(train_cls_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_cls_loader = DataLoader(test_cls_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
train_seg_loader = DataLoader(train_seg_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_seg_loader = DataLoader(test_seg_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ----- TRAINING -----
model = MultiTaskModel().to(DEVICE)
old_model_ckpt = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
model.backbone.load_state_dict({k.replace("backbone.", ""): v for k, v in old_model_ckpt.items() if "backbone" in k}, strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion_cls = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # ---- Classification Training ----
    model.train()
    cls_loss_total = 0
    for img, dr_label, edema_label in tqdm(train_cls_loader, desc="Cls Train"):
        img, dr_label, edema_label = img.to(DEVICE), dr_label.to(DEVICE), edema_label.to(DEVICE)

        dr_pred, edema_pred, _ = model(img)
        loss = criterion_cls(dr_pred, dr_label) + criterion_cls(edema_pred, edema_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cls_loss_total += loss.item()
    
    optimizer.zero_grad(set_to_none=True)  # Better for memory
    torch.cuda.empty_cache()

    # ---- Segmentation Training ----
    seg_loss_total = 0
    for img, mask in tqdm(train_seg_loader, desc="Seg Train"):
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        _, _, seg_pred = model(img)
        loss = dice_loss(seg_pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        seg_loss_total += loss.item()

    print(f"Cls Loss: {cls_loss_total / len(train_cls_loader):.4f} | Seg Loss: {seg_loss_total / len(train_seg_loader):.4f}")

    torch.cuda.empty_cache()

    # ---- Validation on Classification ----
    model.eval()
    all_dr_preds, all_edema_preds = [], []
    all_dr_labels, all_edema_labels = [], []

    with torch.no_grad():
        for img, dr_label, edema_label in tqdm(test_cls_loader, desc="Cls Val"):
            img = img.to(DEVICE)
            dr_label, edema_label = dr_label.to(DEVICE), edema_label.to(DEVICE)

            dr_pred, edema_pred, _ = model(img)
            all_dr_preds.extend(dr_pred.argmax(1).cpu().numpy())
            all_edema_preds.extend(edema_pred.argmax(1).cpu().numpy())
            all_dr_labels.extend(dr_label.cpu().numpy())
            all_edema_labels.extend(edema_label.cpu().numpy())

    acc_dr = accuracy_score(all_dr_labels, all_dr_preds)
    acc_edema = accuracy_score(all_edema_labels, all_edema_preds)
    kappa = cohen_kappa_score(all_dr_labels, all_dr_preds, weights='quadratic')

    print(f"DR Accuracy: {acc_dr:.4f} | Edema Accuracy: {acc_edema:.4f} | Kappa: {kappa:.4f}")

    torch.save(model.state_dict(), f'finetuned_epoch_{epoch+1}.pth')