import os
import pandas as pd
from PIL import Image
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Configuration ---
csv_path = "./data/trainLabels.csv"
image_folder = os.path.join('.', '../../../Downloads/train/')
output_h5_path = "dataset.h5"
resize_shape = (224, 224)  # or (728, 728)
train_ratio = 0.8
delete_images = True  # Set to True to delete original images after saving

# --- Load and prepare DataFrame ---
df = pd.read_csv(csv_path)
df['path'] = df['image'].map(lambda x: os.path.join(image_folder, f"{x}.jpeg"))
df['exists'] = df['path'].map(os.path.exists)
df = df[df['exists']]
df['level'] = df['level'].astype(int)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Split into Train and Validation ---
train_df, val_df = train_test_split(df, test_size=1 - train_ratio, stratify=df['level'], random_state=42)

def create_group(h5_file, group_name, df_subset):
    num_samples = len(df_subset)
    grp = h5_file.create_group(group_name)
    grp.create_dataset("images", shape=(num_samples, *resize_shape, 3), dtype=np.uint8)
    grp.create_dataset("labels", shape=(num_samples,), dtype=int)
    grp.create_dataset("names", shape=(num_samples,), dtype=h5py.string_dtype())

    for idx, (_, row) in enumerate(tqdm(df_subset.iterrows(), total=num_samples, desc=f"Saving '{group_name}'")):
        try:
            with Image.open(row['path']) as im:
                img = im.convert("RGB").resize(resize_shape)
                grp["images"][idx] = np.array(img)
                grp["labels"][idx] = row['level']
                grp["names"][idx] = row['image']

            if delete_images:
                os.remove(row['path'])
        except Exception as e:
            print(f"❌ Error at index {idx} for image {row['path']} — {e}")

# --- Create HDF5 File ---
print("Creating HDF5 file with train and validation sets...")
with h5py.File(output_h5_path, "w") as h5f:
    create_group(h5f, "train", train_df)
    create_group(h5f, "val", val_df)

print(f"\n✅ HDF5 dataset saved to: {output_h5_path}")
