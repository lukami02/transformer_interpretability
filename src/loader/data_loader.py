import os
import json
import logging
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from .bbox_helper import download_imagenet_bbox_annotations, _build_xml_index, parse_voc_xml, get_imagenet_label_map


logger = logging.getLogger("DataPipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


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
    
    bbox_dir = download_imagenet_bbox_annotations(output_dir, logger)
    xml_index = _build_xml_index(bbox_dir, logger)

    logger.info("Initializing Hugging Face dataset stream...")
    os.makedirs(images_dir, exist_ok=True)

    dataset = load_dataset(
        "visual-layer/imagenet-1k-vl-enriched",
        split="validation",
        streaming=True,
    )
    shuffled_dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    subset = shuffled_dataset.take(num_samples)

    annotations: Dict[str, dict] = {}
    no_bbox_count = 0

    logger.info(f"Streaming and persisting {num_samples} samples to disk...")

    for idx, item in enumerate(subset):
        image_id = f"img_{idx:04d}"
        image_path = os.path.join(images_dir, f"{image_id}.jpg")

        img: Image.Image = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(image_path)

        ilsvrc_key = item["image_id"]

        xml_path = xml_index.get(ilsvrc_key)
        bboxes_xyxy: List[Dict] = []

        if xml_path:
            bboxes_xyxy = parse_voc_xml(xml_path)
        else:
            no_bbox_count += 1

        annotations[image_id] = {
            "label_id": item["label"],
            "ilsvrc_key": ilsvrc_key,
            "bboxes": bboxes_xyxy,
        }

        if (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
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
        num_samples: int = 500,
        seed: int = 42
    ):
        download_imagenet_subset(
            output_dir=output_dir,
            num_samples=num_samples,
            seed=seed
        )

        self.images_dir = os.path.join(output_dir, "images")
        self.annotations_file = os.path.join(output_dir, "annotations.json")
        self.transform = transform if transform is not None else _EVAL_TRANSFORM

        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"Annotations missing at '{self.annotations_file}'. Run ingestion first.")

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        self.image_ids = sorted(self.annotations.keys())
        self.label_map = get_imagenet_label_map(output_dir, logger)

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
    
    @staticmethod
    def _bbox_area(box: List[float]) -> float:
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
            best = max(raw_boxes, key=lambda b: self._bbox_area(b["bbox"]))
            xmin, ymin, xmax, ymax = best["bbox"]
    
            scale = self._resize_size / min(orig_w, orig_h)
            new_w = round(orig_w * scale)
            new_h = round(orig_h * scale)

            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
    
            crop_offset_x = (new_w - self._crop_size) / 2.0
            crop_offset_y = (new_h - self._crop_size) / 2.0
    
            xmin = round(xmin * scale_x - crop_offset_x)
            ymin = round(ymin * scale_y - crop_offset_y)
            xmax = round(xmax * scale_x - crop_offset_x)
            ymax = round(ymax * scale_y - crop_offset_y)
    
            xmin = max(0, min(xmin, self._crop_size))
            ymin = max(0, min(ymin, self._crop_size))
            xmax = max(0, min(xmax, self._crop_size))
            ymax = max(0, min(ymax, self._crop_size))
    
            scaled_bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
        else:
            scaled_bbox = torch.tensor([-1, -1, -1, -1], dtype=torch.float32)
    
        return img_tensor, label, scaled_bbox
    
def get_xai_dataloader(
    output_dir: str = "./imagenet_xai_subset",
    transform: Optional[transforms.Compose] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    num_samples: int = 5000,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Utility function to get DataLoader for XAI evaluation.
    """
    dataset = LocalImageNetXAIDataset(output_dir=output_dir, transform=transform, num_samples=num_samples, seed=seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False,
    )