# 📌 Project Objectives

# 1. Damage Localization – Detect where the damage is on the vehicle using bounding boxes.

# 2. Component Identification – Identify the specific part of the car that is damaged.

# 3. Damage Classification – Label the damage type (e.g., dent, scratch, broken glass).

# 4. Dataset Fusion – Combine multiple datasets (CarDD, Car Parts & Damages dataset, COCO-based sets) for stronger generalization.

# 5. Evaluation – Assess model performance using standard metrics (mAP, precision, recall, F1-score).

# 6. Deployment Ready – Provide an inference pipeline (API or app) where users can upload images to detect damages.


import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class VehicleDamageDataset(Dataset):
    def __init__(self, img_dir, annotation_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(annotation_file) as f:
            self.coco = json.load(f)
        self.imgs = {img['id']: img for img in self.coco['images']}
        self.anns = self.coco['annotations']

        # Map image_id to annotations
        self.img_to_anns = {}
        for ann in self.anns:
            self.img_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.ids = list(self.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels,
                  'image_id': torch.tensor([img_id])}

        if self.transforms:
            img = self.transforms(img)

        return img, target
