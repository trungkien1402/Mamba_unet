import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random


class ToothDataset(Dataset):

    def __init__(self, root_dir, split="train", img_size=512, augment=True):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "img")
        self.mask_dir = os.path.join(root_dir, "masks_machine")

        self.img_size = img_size
        self.split = split

        # =============================
        # 🔥 DANH SÁCH ẢNH RĂNG GÃY
        # =============================
        self.broken_files = {
            "16.png", "20.png", "22.png", "24.png",
            "607.png", "622.png", "637.png",
            "867.png", "868.png", "869.png", "870.png",
            "871.png", "872.png", "873.png", "874.png",
            "875.png", "876.png", "877.png", "878.png",
            "879.png", "880.png", "881.png", "882.png",
            "883.png", "884.png", "885.png", "886.png",
            "887.png", "888.png", "889.png",
            "890.jpg",
        }

        # =============================
        # LOAD FILE LIST
        # =============================
        all_images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        random.seed(42)
        random.shuffle(all_images)

        # =============================
        # SPLIT 80 / 10 / 10
        # =============================
        n_total = len(all_images)
        n_train = int(0.8 * n_total)

        if split == "train":
            self.image_list = all_images[:n_train]

        elif split == "val":
            # Lấy toàn bộ phần còn lại (tương đương 20%)
            self.image_list = all_images[n_train:]

        else:
            raise ValueError("split must be 'train' or 'val'")

        # =============================
        # SAMPLE WEIGHTS (train only)
        # =============================
        if split == "train":
            self.sample_weights = [
                4.0 if img in self.broken_files else 1.0
                for img in self.image_list
            ]
        else:
            self.sample_weights = None

        # =============================
        # AUGMENTATION
        # =============================
        if augment and split == "train":

            self.transform = A.Compose([
                A.Resize(img_size, img_size),

                A.HorizontalFlip(p=0.5),

                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.6
                ),

                A.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.25,
                    p=0.5
                ),

                A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),

                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),

                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])

        else:
            # val/test → không augment random
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])

        # =============================
        # LOG
        # =============================
        print(f"✓ Loaded {len(self.image_list)} {split} samples")

        if split == "train":
            n_broken = sum(
                1 for img in self.image_list
                if img in self.broken_files
            )
            print(f"✓ Broken samples in train: {n_broken}")

    # =====================================
    # LENGTH
    # =====================================
    def __len__(self):
        return len(self.image_list)

    # =====================================
    # GET ITEM
    # =====================================
    def __getitem__(self, idx):

        img_name = self.image_list[idx]
        mask_name = os.path.splitext(img_name)[0] + ".png"

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # ---- Load grayscale ----
        image = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        # ---- Binary mask ----
        mask = (mask > 0).astype(np.uint8)

        transformed = self.transform(image=image, mask=mask)

        image = transformed["image"]   # (1, H, W)
        mask = transformed["mask"]     # (H, W)

        # đảm bảo có channel
        if image.ndim == 2:
            image = image.unsqueeze(0)

        return image.float(), mask.long()