import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def main():
    # Load a YOLO model
    model = YOLO("LidTest.pt")

    # Run inference on a test image
    results = model("camera_rgb-4806-633428505.png", save=True)

    

if __name__ == '__main__':
    main()