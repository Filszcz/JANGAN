import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from timeit import default_timer as timer

def calculate_dataset_statistics(dataset_path):
    """
    Calculate mean and standard deviation for a dataset of RGB images.
    
    Args:
        dataset_path (str): Path to directory containing images
        
    Returns:
        tuple: (mean_per_channel, std_per_channel)
    """
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Get list of all image files
    image_files = [
        os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not image_files:
        raise ValueError("No valid images found in the specified directory")

    # Initialize arrays to store sums and squared sums
    sum_per_channel = np.zeros(3)
    squared_sum_per_channel = np.zeros(3)
    pixel_count = 0

    # Process each image
    print("Processing images...")
    for img_path in tqdm(image_files):
        try:
            # Open image and convert to RGB
            img = Image.open(img_path).convert('RGB')
            # Convert to numpy array and reshape to (num_pixels, 3)
            img_array = np.array(img).reshape(-1, 3)
            
            # Update sums
            sum_per_channel += img_array.sum(axis=0)
            squared_sum_per_channel += (img_array ** 2).sum(axis=0)
            pixel_count += img_array.shape[0]
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

    # Calculate mean and standard deviation
    mean_per_channel = sum_per_channel / pixel_count
    std_per_channel = np.sqrt(
        ((squared_sum_per_channel / pixel_count) - (mean_per_channel))**2
    )

    return mean_per_channel, std_per_channel

# Example usage
if __name__ == "__main__":
    dataset_path = "archive"
    start_time = timer()
    means, stds = calculate_dataset_statistics(dataset_path)
    end_time = timer()
   
    print("\nDataset Statistics:")
    print(f"Means: R={means[0]:.3f}, G={means[1]:.3f}, B={means[2]:.3f}")
    print(f"Stds: R={stds[0]:.3f}, G={stds[1]:.3f}, B={stds[2]:.3f}")
    print(f"Time taken: {end_time-start_time}s")







