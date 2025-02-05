import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import uuid



# Load the RGB image
rgb_image = Image.open("bus.jpg").convert("RGB")

# Load the depth map (assuming it's a grayscale image)
depth_map = Image.open("bus_depth.png").convert("L")

# Convert images to numpy arrays
rgb_array = np.array(rgb_image)
depth_array = np.array(depth_map)

# Combine RGB and depth into an RGBA image
rgba_array = np.dstack((rgb_array, depth_array))

# Convert back to PIL Image
rgba_image = Image.fromarray(rgba_array, 'RGBA')

# Save the combined image for visualization 

# Generate a unique filename
unique_filename = f"bus_rgba_{uuid.uuid4().hex}.png"

# Ensure the directory exists
os.makedirs("datasets/RGBDdataset", exist_ok=True)

# Save the combined image with the unique filename
rgba_image.save(f"datasets/RGBDdataset/{unique_filename}")


# Convert the RGBA image to a tensor
# rgba_tensor = torch.from_numpy(np.array(rgba_image)).permute(2, 0, 1).float().unsqueeze(0)

# np.array(rgba_image): Converts the PIL image rgba_image to a NumPy array. The resulting array has the shape (height, width, 4), where the last dimension represents the RGBA channels.
# torch.from_numpy(...): Converts the NumPy array to a PyTorch tensor. The shape of the tensor remains (height, width, 4).
# .permute(2, 0, 1): Changes the order of dimensions of the tensor. The permute function rearranges the dimensions to (4, height, width). This is necessary because PyTorch models typically expect tensors in the shape (channels, height, width).
# .float(): Converts the tensor to a floating-point type. This is often required for input tensors to neural networks, as many models expect floating-point inputs.
# .unsqueeze(0):Adds an extra dimension at the 0th position, changing the shape from (4, height, width) to (1, 4, height, width). This extra dimension represents the batch size, which is required for most PyTorch models. In this case, the batch size is 1.
# Putting it all together, this line of code converts the RGBA image into a PyTorch tensor with the shape (1, 4, height, width), suitable for input into a neural network.

# Run inference with the YOLO11n model on the RGBA image
# results = model(rgba_tensor, save=True)

# Display the image using matplotlib to keep it open
plt.imshow(rgba_image)
plt.show()