# preprocess_and_save.py
import h5py
import numpy as np
import cv2
from PIL import Image
import os

IMAGE_SIZE = 380
HDF5_PATH = './data/eyepacs.h5'
OUT_DIR = './data/preprocessed'

os.makedirs(OUT_DIR, exist_ok=True)

def resize_with_padding(img, desired_size=IMAGE_SIZE):
    old_size = img.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)

    # If bounding box is invalid, skip cropping
    if w == 0 or h == 0:
        cropped = img
    else:
        cropped = img[y:y+h, x:x+w]

    return resize_with_padding(cropped, IMAGE_SIZE)


with h5py.File(HDF5_PATH, 'r') as f:
    for split in ['train', 'val']:
        raw_imgs = f[f'{split}/images'][:]
        labels = f[f'{split}/labels'][:]
        print(f"Processing {split} set: {len(raw_imgs)} images")

        processed_imgs = []
        for img in raw_imgs:
            processed = preprocess_image(img)
            processed_imgs.append(processed)

        processed_imgs = np.stack(processed_imgs)
        np.save(os.path.join(OUT_DIR, f'{split}_images.npy'), processed_imgs)
        np.save(os.path.join(OUT_DIR, f'{split}_labels.npy'), labels)
        print(f"Saved {split}_images.npy and {split}_labels.npy")
