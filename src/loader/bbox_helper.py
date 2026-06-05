import os
import json
import logging
import tarfile
import urllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset


BBOX_TAR_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz"
BBOX_TAR_FILENAME = "ILSVRC2012_bbox_val_v3.tgz"


def download_imagenet_bbox_annotations(output_dir: str, logger: logging.Logger = None) -> str:
    bbox_dir = os.path.join(output_dir, "bbox_annotations")
    if os.path.isdir(bbox_dir) and any(Path(bbox_dir).rglob("*.xml")):
        if logger:
            logger.info(f"Bbox annotations already extracted at '{bbox_dir}'. Skipping download.")
        return bbox_dir

    tar_path = os.path.join(output_dir, BBOX_TAR_FILENAME)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(tar_path):
        if logger:
            logger.info(f"Downloading bbox annotations from {BBOX_TAR_URL} ...")

        urllib.request.urlretrieve(BBOX_TAR_URL, tar_path)

    if logger:
        logger.info(f"Extracting '{tar_path}' ...")
    os.makedirs(bbox_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(bbox_dir)

    logger.info(f"Bbox annotations extracted to '{bbox_dir}'.")
    return bbox_dir


def parse_voc_xml(xml_path: str) -> List[Dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = int(float(bndbox.findtext("xmin")))
            ymin = int(float(bndbox.findtext("ymin")))
            xmax = int(float(bndbox.findtext("xmax")))
            ymax = int(float(bndbox.findtext("ymax")))
        except (TypeError, ValueError):
            continue
        boxes.append({"label": name, "bbox": [xmin, ymin, xmax, ymax]})
    return boxes


def _build_xml_index(bbox_dir: str, logger: logging.Logger = None) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for xml_path in Path(bbox_dir).rglob("*.xml"):
        index[xml_path.stem] = str(xml_path)
    logger.info(f"Indexed {len(index)} XML annotation files.")
    return index

def get_imagenet_label_map(output_dir: str, logger: logging.Logger = None) -> Dict[int, str]:
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
