# HHA stands for Horizontal disparity, Height above ground, and Angle with gravity. This script takes in a depth image and outputs an HHA image.

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import os
# Load all images from the combinator/input directory -- only pull odd images
input_dir = "combinator/input"
images = [Image.open(os.path.join(input_dir, file)) for file in os.listdir(input_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
for i in range(1, len(images), 2):
    # Load the depth map (assuming it's a grayscale image)
    depth_map = images[i].convert("L")
    # Invert the depth map - closer values are darker
    depth_map = ImageOps.invert(depth_map)
    # Normalize the depth map to ensure values are in the range [0, 255]
    depth_map = np.array(depth_map)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    depth_map = depth_map.astype(np.uint8)
    # Convert images to numpy arrays
    depth_array = np.array(depth_map)
    # Calculate horizontal disparity
    h_disp = np.abs(depth_array[:, :-1].astype(np.float32) - depth_array[:, 1:].astype(np.float32))
    # Calculate height above ground
    h_above_ground = np.zeros_like(depth_array)
    for row in range(1, depth_array.shape[0]):
        h_above_ground[row] = h_above_ground[row - 1] + (depth_array[row - 1] + depth_array[row]) / 2
    # Calculate angle with gravity
    angle = np.arctan(h_disp / h_above_ground)
    # Normalize angle to [0, 255]
    angle = (angle - angle.min()) / (angle.max() - angle.min()) * 255
    # Convert to uint8
    angle = angle.astype(np.uint8)
    # Combine H, H, and A into an HHA image
    hha_array = np.dstack((h_disp, h_above_ground, angle))
    # Convert back to PIL Image
    hha_image = Image.fromarray(hha_array, 'RGB')
    # Save the HHA image
    filename = f"hha_{i}.png"
    os.makedirs("combinator/output", exist_ok=True)
    hha_image.save(f"combinator/output/{filename}")
    # Display the HHA image using matplotlib
    plt.imshow(hha_image)
    plt.show()
    # Save the HHA image with the unique filename
    hha_image.save(f"combinator/output/{filename}")
    # Display the HHA image using matplotlib to keep it open
    plt.imshow(hha_image)
    plt.show()