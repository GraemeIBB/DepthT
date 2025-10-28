import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

def main():
    # Load a YOLO model
    model = YOLO("yolo11n.yaml")

    # Train the model using your custom dataset (update the data file accordingly)
    results = model.train(
        data="RGBLid.yaml", 
        epochs=100, 
        imgsz=640,  
        save_period=1,
        project="runs/train",
        name="exp",
        workers=2,
        batch=16,
        lr0=0.01,
        augment=True,
    )

    # Run inference on a test image
    # results = model("camera_rgb-4886-27716661.png", save=True, show=True)
    
    # Save the final weights
    import shutil
    shutil.copy("runs/detect/train/weights/last.pt", "models/LidTest.pt")

if __name__ == '__main__':
    main()
