import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CloudDataset(Dataset):
    def __init__(self, base_dir="data/38-Cloud_training"):
        self.base_dir = base_dir
        self.gt_dir = os.path.join(base_dir, "train_gt")
        self.gt_files = [
            f for f in os.listdir(self.gt_dir) if f.lower().endswith(".tif")
        ]

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt_name = self.gt_files[idx]
        patch_id = gt_name.replace("gt_", "")

        bands = ["red", "green", "blue", "nir"]
        channels = []
        for band in bands:
            path = os.path.join(self.base_dir, f"train_{band}", f"{band}_{patch_id}")
            arr = np.array(Image.open(path)).astype(np.float32)
            min_val, max_val = arr.min(), arr.max()
            if max_val - min_val > 0:
                arr = (arr - min_val) / (max_val - min_val)
            channels.append(arr)

        img = np.stack(channels, axis=0)

        gt_path = os.path.join(self.gt_dir, gt_name)
        mask = np.array(Image.open(gt_path)).astype(np.float32)
        mask = np.where(mask > 0, 1.0, 0.0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img), torch.tensor(mask)