import os
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset

logger = logging.getLogger("DataPipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def get_imagenet_label_map(output_dir: str) -> Dict[int, str]:
    """
    Utility function to get mapping from ImageNet label IDs to human-readable names.
    """
    label_map_path = os.path.join(output_dir, "imagenet_label_map.json")

    if os.path.exists(label_map_path):
        logger.info("Loading cached ImageNet label map...")
        with open(label_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    
    logger.info("Fetching ImageNet label map from dataset features...")
    os.makedirs(output_dir, exist_ok=True)
 
    dataset = load_dataset(
        "visual-layer/imagenet-1k-vl-enriched",
        split="validation",
        streaming=True
    )
    label_names = dataset.features["label"].names
    idx_to_label = {i: name for i, name in enumerate(label_names)}
 
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, indent=2, ensure_ascii=False)
 
    logger.info(f"Label map saved to '{label_map_path}' ({len(idx_to_label)} classes).")
    return idx_to_label

def download_imagenet_subset(
    output_dir: str = "./imagenet_xai_subset",
    num_samples: int = 5000,
    seed: int = 42,
    buffer_size: int = 10000
) -> None:
    """
    Downloads a subset of ImageNet-1K validation data via streaming.
    """
    images_dir = os.path.join(output_dir, "images")
    annotations_file = os.path.join(output_dir, "annotations.json")

    if os.path.exists(images_dir) and os.path.exists(annotations_file):
        logger.info(f"Local cache found at '{output_dir}'. Skipping download phase.")
        return

    logger.info("Initializing Hugging Face dataset stream...")
    os.makedirs(images_dir, exist_ok=True)

    idx_to_label = get_imagenet_label_map(output_dir)
    
    dataset = load_dataset("visual-layer/imagenet-1k-vl-enriched", split="validation", streaming=True)

    shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    subset = shuffled_dataset.take(num_samples)
    
    annotations = {}
    no_bbox_count = 0
    
    logger.info(f"Streaming and persisting {num_samples} samples to disk...")

    for idx, item in enumerate(subset):
        image_id = f"img_{idx:04d}"
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(image_path)

        label_idx = item.get("label")
        gt_label_name = idx_to_label.get(label_idx, "").lower()

        bboxes_xyxy = []
        for b in item.get("label_bbox_enriched", []):
            xmin, ymin, w, h = b["bbox"]
            bboxes_xyxy.append({ 
                    "bbox": [xmin, ymin, xmin + w, ymin + h] ,
                    "confidence": b.get("confidence", 0.0), 
                    "label": b.get("label", "")
            })
 
        if not bboxes_xyxy:
            no_bbox_count += 1
        
        annotations[image_id] = {
            "label_id": label_idx,
            "label_name": idx_to_label.get(label_idx, "") if label_idx is not None else "",
            "bboxes": bboxes_xyxy,
        }

        if (idx + 1) % 500 == 0:
            logger.info(f"  {idx + 1}/{num_samples} samples processed...")

    with open(annotations_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)

    logger.info(
        f"Data ingestion complete. Assets saved to: '{output_dir}'\n"
        f"  Samples with no matching bbox: {no_bbox_count}/{num_samples} "
        f"({100 * no_bbox_count / num_samples:.1f}%)"
    )


class LocalImageNetXAIDataset(Dataset):
    """
    Custom Dataset designed for XAI evaluation.
    """
    def __init__(
        self, 
        output_dir: str, 
        transform: Optional[transforms.Compose] = None,
        confidence_threshold: float = 0.5
    ):
        

        self.images_dir = os.path.join(output_dir, "images")
        self.annotations_file = os.path.join(output_dir, "annotations.json")
        self.transform = transform if transform is not None else _EVAL_TRANSFORM
        self.confidence_threshold = confidence_threshold

        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"Annotations missing at {self.annotations_file}. Run ingestion first.")

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        
        self.image_ids = sorted(list(self.annotations.keys()))

        self._resize_size = 256
        self._crop_size = 224
        for t in self.transform.transforms:
            if isinstance(t, transforms.Resize):
                s = t.size
                self._resize_size = s if isinstance(s, int) else s[0]
            elif isinstance(t, transforms.CenterCrop):
                s = t.size
                self._crop_size = s if isinstance(s, int) else s[0]

    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _bbox_area(self, box: List[float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        
        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        img_tensor = self.transform(pil_img)
        
        anno = self.annotations[img_id]
        label = anno["label_id"]
        raw_boxes = anno.get("bboxes", [])
        
        if raw_boxes:
            confident_boxes = [b for b in raw_boxes if b.get("confidence", 0.0) >= self.confidence_threshold]
            candidates = confident_boxes if confident_boxes else [max(raw_boxes, key=lambda b: b.get("confidence", 0.0))]

            best = max(candidates, key=self._bbox_area)
            xmin, ymin, xmax, ymax = best["bbox"]

            if orig_w < orig_h:
                scale_x = float(self._resize_size) / orig_w
                scale_y = scale_x
            else:
                scale_y = float(self._resize_size) / orig_h
                scale_x = scale_y
            crop_offset_x = (math.ceil(orig_w * scale_x) - self._crop_size) // 2
            crop_offset_y = (math.ceil(orig_h * scale_y) - self._crop_size) // 2      

            xmin = int(xmin * scale_x) - crop_offset_x
            ymin = int(ymin * scale_y) - crop_offset_y
            xmax = int(xmax * scale_x) - crop_offset_x
            ymax = int(ymax * scale_y) - crop_offset_y

            xmin = max(0, min(xmin, self._crop_size))
            ymin = max(0, min(ymin, self._crop_size))
            xmax = max(0, min(xmax, self._crop_size))
            ymax = max(0, min(ymax, self._crop_size))
            
            scaled_bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
        else:
            scaled_bbox = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32)

        return img_tensor, label, scaled_bbox


def get_xai_dataloader(
    output_dir: str = "./imagenet_xai_subset",
    transform: Optional[transforms.Compose] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    num_samples: int = 5000, 
    confidence_threshold = 0.5,
    pin_memory: bool = True
) -> DataLoader:
    """
    Utility function to get DataLoader for XAI evaluation.
    """
    download_imagenet_subset(output_dir=output_dir, num_samples=num_samples, seed=seed)
    
    dataset = LocalImageNetXAIDataset(output_dir=output_dir, transform=transform, confidence_threshold=confidence_threshold)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False
    )