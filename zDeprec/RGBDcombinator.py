# Depreciated: 2025-03-04

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import os

# Load all images from the combinator/input directory -- even is rgb, odd is depth, a pair is (even, even+1)
input_dir = "combinator/input"
images = [Image.open(os.path.join(input_dir, file)) for file in os.listdir(input_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Assuming you want to process the first image in the list
rgb_image = images[0].convert("RGB")

# Load the depth map (assuming it's a grayscale image)
depth_map = images[1].convert("L")

# Invert the depth map - closer values are darker
depth_map = ImageOps.invert(depth_map)

# Normalize the depth map to ensure values are in the range [0, 255]
depth_map = np.array(depth_map)
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
depth_map = depth_map.astype(np.uint8)

# Convert images to numpy arrays
rgb_array = np.array(rgb_image)

# Combine RGB and depth into an RGBA image
rgba_array = np.dstack((rgb_array, depth_map))

# Convert back to PIL Image
rgba_image = Image.fromarray(rgba_array, 'RGBA')

# Save the combined image for visualization 

# Generate a unique filename
filename = "rgba.png"

# Ensure the directory exists
os.makedirs("combinator/output", exist_ok=True)

# Save the combined image with the unique filename
rgba_image.save(f"combinator/output/{filename}")

# Display the image using matplotlib to keep it open
plt.imshow(rgba_image)
plt.show()