# HHA stands for Horizontal disparity, Height above ground, and Angle with gravity. This script takes in a depth image and outputs an HHA image.
import cv2
import numpy as np
import os

# Camera intrinsics
INTRINSICS = np.array([
    [389.7704162597656, 0, 319.655517578125],
    [0, 389.7704162597656, 236.47743225097656],
    [0, 0, 1]
], dtype=np.float32)

def depth_to_hha(depth, intrinsics):
    """
    Parameters:
    - depth (numpy array): Input depth image (floating-point in meters).
    - intrinsics (numpy array): 3x3 camera intrinsic matrix.

    Returns:
    - HHA image (numpy array)
    """
    # Convert depth to float32
    depth = depth.astype(np.float32)
    # Scaling factor
    depth /=10.0
    
    # Handle missing depth values (treat black pixels as very close)
    missing_mask = (depth <= 0)
    depth[missing_mask] = 0.01  # Small value to avoid division errors

    # Extract camera intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Image dimensions
    h, w = depth.shape[:2]  # Use only the first two dimensions, third dimension is for channels

    # Create coordinate grid
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Compute 3D coordinates (X, Y, Z)
    Z = depth
    X = (u_coords - cx) * Z / fx
    Y = (v_coords - cy) * Z / fy
    point_cloud = np.dstack((X, Y, Z))

    # Estimate up direction (gravity vector)
    yDir = np.array([0, -1, 0])

    # Compute height above ground
    proj_on_up = (point_cloud @ yDir)  # Dot product with up vector
    camera_height = -np.min(proj_on_up) if np.any(Z > 0) else 0
    height = camera_height + proj_on_up
    height[height < 0] = 0  # Clamp negative heights

    # Compute angle with respect to gravity
    angle = np.degrees(np.arccos(np.clip(Y / np.maximum(np.sqrt(X**2 + Y**2 + Z**2), 1e-6), -1, 1)))
    angle[missing_mask] = 180  # Default angle for missing pixels

    # Compute horizontal disparity (clamping min depth to 1m as per Gupta et al.)
    Z_clamped = np.maximum(Z * 100.0, 100.0)  # Convert Z to cm, min depth 100cm
    disparity = 31000.0 / Z_clamped

    # Prepare HHA channels
    I1 = disparity  # Disparity
    I2 = height  # Height above ground
    I3 = angle + 38  # Adjust angle encoding

    # Stack and normalize
    HHA = np.dstack((I1, I2, I3))
    HHA = np.clip(np.rint(HHA), 0, 255).astype(np.uint8)  # Convert to 8-bit

    return HHA

def process_folder(input_folder, output_folder, intrinsics):
    """
    Processes all PNG depth images in a folder and saves corresponding HHA images.

    Parameters:
    - input_folder (str): Path to folder containing depth images.
    - output_folder (str): Path to save HHA images.
    - intrinsics (numpy array): 3x3 camera intrinsic matrix.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all PNG files in input folder
    depth_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    for depth_file in depth_files:
        input_path = os.path.join(input_folder, depth_file)
        output_path = os.path.join(output_folder, depth_file.replace('.png', '_hha.png'))

        print(f"Processing {input_path} -> {output_path}")

        # Read depth image (unchanged to preserve precision)
        depth = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            print(f"Warning: Unable to read {input_path}, skipping.")
            continue

        # Ensure depth image has only two dimensions
        if len(depth.shape) > 2:
            depth = depth[:, :, 0]  # Extract the first channel

        # Convert to HHA
        HHA = depth_to_hha(depth, intrinsics)

        # Save output
        cv2.imwrite(output_path, HHA)

    print(f"Processing complete! HHA images saved in {output_folder}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch convert depth images to HHA format.")
    parser.add_argument("--input_folder", "-i", required=True, help="Path to input folder containing depth PNG images.")
    parser.add_argument("--output_folder", "-o", required=True, help="Path to output folder for HHA images.")
    
    args = parser.parse_args()

    # Run batch processing
    process_folder(args.input_folder, args.output_folder, INTRINSICS)
