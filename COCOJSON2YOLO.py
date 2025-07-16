import json
import os
from collections import defaultdict

def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_annotation(ann, image_w, image_h):
    category_id = ann['category_id'] - 1  # YOLO expects 0-based class ids
    bbox = ann['bbox']
    seg = ann['segmentation'][0]  # assume 1 polygon only

    # Convert bbox
    x, y, w, h = bbox
    xc = (x + w / 2) / image_w
    yc = (y + h / 2) / image_h
    bw = w / image_w
    bh = h / image_h

    # Convert segmentation
    seg_norm = [str(coord / image_w if i % 2 == 0 else coord / image_h) for i, coord in enumerate(seg)]

    return f"{category_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} " + " ".join(seg_norm)

def coco_to_yolo(json_path, output_dir):
    coco_data = load_json(json_path)
    os.makedirs(output_dir, exist_ok=True)
    # Build a mapping from image_id to (file_name, width, height)
    image_id_to_info = {img['id']: (img['file_name'], img['width'], img['height']) for img in coco_data['images']}
    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        anns_by_image[ann['image_id']].append(ann)
    # Write YOLO txt files
    for image_id, anns in anns_by_image.items():
        file_name, image_w, image_h = image_id_to_info[image_id]
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, 'w') as f:
            for ann in anns:
                yolo_line = convert_annotation(ann, image_w, image_h)
                f.write(yolo_line + '\n')

# Example usage:
coco_to_yolo('datasets/simseg/1-30.json', 'datasets/simseg/labels/val')

