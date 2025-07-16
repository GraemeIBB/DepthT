import torch
import cv2
from ultralytics import YOLO

# Set the device to the cuda GPU (index 0) if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Load a COCO-pretrained YOLO11n model and move it to the specified device
model = YOLO("yolo11n.pt").to(device)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to a tensor, normalize it, and move it to the specified device
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    # Run YOLO inference on the frame with show=True to display the results
    results = model(frame_tensor, show=True)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()