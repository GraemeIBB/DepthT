import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from PIL import Image
import os

class CocoDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms if transforms else T.ToTensor()

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        return self.transforms(img), target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    # === Settings ===
    img_dir = "datasets/gate_dataset/image/train"              # <-- Change this
    ann_file = "path/to/annotations.json"   # <-- Change this
    num_classes = 2  # 1 class + background
    num_epochs = 10
    batch_size = 2
    lr = 0.005

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load Dataset ===
    dataset = CocoDetectionDataset(img_dir, ann_file)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn)

    # === Model ===
    model = get_model(num_classes)
    model.to(device)

    # === Optimizer ===
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # === Training Loop ===
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # === Save Model ===
    os.makedirs("models", exist_ok=True)  # Create the models folder if it doesn't exist
    model_path = os.path.join("models", "faster_rcnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
