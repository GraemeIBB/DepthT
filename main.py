import torch
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.yaml")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(
    data="RGBLid.yaml", 
    epochs=100, 
    imgsz=[640, 480],  # Resize images to 640x480
    save_period=1,  # Save the model weights every epoch
    project="runs/train",  # Directory to save the training results
    name="exp",  # Name of the experiment
    workers=2,  
    rect=True,
)

# Run inference with the YOLO11n model on the test image
results = model("camera_rgb-4886-27716661.png", save=True)

# Save the final model weights with a specific name
import shutil
shutil.copy("runs/detect/train/weights/last.pt", "LidTest.pt")
