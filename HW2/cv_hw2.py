# -*- coding: utf-8 -*-
"""cv_hw2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gjWQ7CRvcDjFyR02MZLAiXG_e1ZPO-SD
"""

from google.colab import drive
drive.mount('/content/drive')

import os, json, tarfile
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F

from torchvision.ops import box_convert
from pycocotools.cocoeval import COCOeval

import torchvision.transforms as T

# ==== [Part 1] decompress & read data ====
DATA_PATH = './nycu-hw2-data'
if not os.path.exists(DATA_PATH):
    with tarfile.open('/content/drive/MyDrive/Colab Notebooks/nycu-hw2-data.tar', 'r') as tar:
        tar.extractall('.')

train_coco = COCO(f'{DATA_PATH}/train.json')
val_coco = COCO(f'{DATA_PATH}/valid.json')

if os.path.exists(DATA_PATH):
    # check train/valid dir
    train_dir = os.path.join(DATA_PATH, 'train')
    valid_dir = os.path.join(DATA_PATH, 'valid')
    if os.path.exists(train_dir) and os.path.exists(valid_dir):
        # check train/valid image num
        train_images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        valid_images = [f for f in os.listdir(valid_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"資料夾 '{train_dir}' 中的 image 數量：{len(train_images)}")
        print(f"資料夾 '{valid_dir}' 中的 image 數量：{len(valid_images)}")
    else:
        print("train 或 valid 資料夾不存在")

# ==== Dataset ====
class COCODigitDataset(Dataset):
    def __init__(self, coco, img_dir):
        self.coco = coco
        self.img_dir = img_dir
        self.ids = list(coco.imgs.keys())

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        image = Image.open(os.path.join(self.img_dir, self.coco.loadImgs(img_id)[0]['file_name'])).convert('RGB')
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        image = F.to_tensor(image)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = COCODigitDataset(train_coco, f'{DATA_PATH}/train')
val_dataset = COCODigitDataset(val_coco, f'{DATA_PATH}/valid')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)

def analyze_bbox_sizes(coco):
    widths, heights = [], []
    for img_id in coco.imgs.keys():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            widths.append(bbox[2])
            heights.append(bbox[3])
    print(f"Average width: {sum(widths)/len(widths):.1f}, Average height: {sum(heights)/len(heights):.1f}")
    print(f"Min width: {min(widths):.1f}, Max width: {max(widths):.1f}")
    print(f"Min height: {min(heights):.1f}, Max height: {max(heights):.1f}")

analyze_bbox_sizes(train_coco)

# ==== [Part 2] build model ====
def build_model(num_classes=11):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,)) #[exp]
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_gen,
                       box_score_thresh=0.05, box_detections_per_img=15)
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ==== [Part 3] trainin & validation ====
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    for imgs, targets in tqdm(loader):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# evaluate with mAP
def evaluate_model(model, coco_gt, loader):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Evaluating"):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)

            for output, target in zip(outputs, targets):
                image_id = int(target['image_id'].item())
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    w, h = x2 - x1, y2 - y1
                    results.append({
                        'image_id': image_id,
                        'category_id': int(label.item()),
                        'bbox': [x1, y1, w, h],
                        'score': float(score)
                    })

    # COCO
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]

# load previous checkpoint
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
start_epoch = 0

checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pth')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {start_epoch}")

num_epochs = 5
for epoch in range(start_epoch, num_epochs):
    loss = train_one_epoch(model, train_loader)
    map_score = evaluate_model(model, val_coco, val_loader)
    print(f"[Epoch {epoch+1}] Loss: {loss:.4f}, mAP: {map_score:.4f}")


    # save checkpoint every 5 epoch
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'check_point_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

# save model
from google.colab import files

torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_model.pth'))
print("Final model saved.")
files.download('./checkpoints/final_model.pth')
files.download(f'./checkpoints/check_point_{epoch + 1}.pth')

# ==== [Part 4] test and output pred.json & pred.csv ====
import os
from PIL import Image
from torchvision.transforms import functional as F
import torch
from tqdm import tqdm
import pandas as pd
import json
from collections import defaultdict

model.eval()

test_img_dir = os.path.join(DATA_PATH, 'test')
test_img_filenames = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.png')])
test_imgs = [{'id': int(f.replace('.png', '')), 'file_name': f} for f in test_img_filenames]

results = []
with torch.no_grad():
    for img in tqdm(test_imgs):
        image = Image.open(os.path.join(test_img_dir, img['file_name'])).convert('RGB')
        tensor = F.to_tensor(image).unsqueeze(0).to(device)
        output = model(tensor)[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score < 0.7:  # tried:0.9, 0.8, 0.7 best:0.7
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append({
                'image_id': img['id'],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(score),
                'category_id': int(label.item())
            })

# save pred.json
results.sort(key=lambda x: x['image_id'])
with open('pred.json', 'w') as f:
    json.dump(results, f)

# Task 2
pred_digits = defaultdict(list)
for r in results:
    x = r['bbox'][0]
    pred_digits[r['image_id']].append((x, r['category_id'] - 1))

rows = []
for img in test_imgs:
    image_id = img['id']
    if image_id not in pred_digits:
        rows.append({'image_id': image_id, 'pred_label': -1})
    else:
        digits = sorted(pred_digits[image_id], key=lambda x: x[0])
        pred_str = ''.join(str(d[1]) for d in digits)
        rows.append({'image_id': image_id, 'pred_label': pred_str})

rows.sort(key=lambda x: x['image_id'])
pd.DataFrame(rows).to_csv('pred.csv', index=False)
print("\npred.json & pred.csv 已輸出完畢")

# download pred.json pred.csv checkpoint.pth

from google.colab import files
files.download('pred.json')
files.download('pred.csv')

# ==== Visualize ====
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_prediction(image_path, boxes, scores, labels, threshold=0.01):
    image = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{label.item()} ({score:.2f})", color='black',
                fontsize=12, bbox=dict(facecolor='lime', alpha=0.5))
    plt.axis('off')
    plt.show()


image_ids = list(range(1, 11))
filtered_test_imgs = [img for img in test_imgs if img['id'] in image_ids]
filtered_test_imgs.sort(key=lambda x: x['id'])

for img in filtered_test_imgs:
    image_path = os.path.join(DATA_PATH, 'test', img['file_name'])
    image_tensor = F.to_tensor(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]

    print(f"🔍 Image: {img['file_name']} | Boxes predicted: {len(output['boxes'])}")
    visualize_prediction(
        image_path,
        output['boxes'].cpu().numpy(),
        output['scores'].cpu().numpy(),
        output['labels'].cpu()
    )