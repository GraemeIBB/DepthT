import torch
import cv2
from ultralytics import YOLO

# Set the device to the first cuda GPU (index 0) if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a COCO-pretrained YOLO11n model and move it to the specified device
model = YOLO("yolo11n.pt").to(device)

# URL of the live feed
url = "https://streamserve.ok.ubc.ca/LiveCams/timcam.stream_720p/playlist.m3u8"

# Initialize video capture from the URL
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the frame to 640x640
    resized_frame = cv2.resize(frame, (640, 640))

    # Convert the frame to a tensor, normalize it, and move it to the specified device
    frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    # Run YOLO inference on the frame with show=True to display the results
    results = model(frame_tensor, show=True)

    # Add a small delay to make the window responsive
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
