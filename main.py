import os
import cv2
import requests

from src.noise import add_gaussian_noise, add_salt_and_pepper_noise
from src.filters import apply_mean_filter, apply_median_filter, apply_gaussian_filter
from src.metrics import calculate_mse, calculate_psnr
from src.visualize import plot_and_save_images, plot_metrics

def download_sample_images(data_dir):
    images = {
        "lena.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "peppers.png": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/peppers.png",
        "baboon.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
        "building.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg"
    }
    for filename, url in images.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {filename}")
    print("Sample images dataset ready.")

def process_noise_pipeline(image, noise_type, noise_func, output_dir, image_name="image"):
    print(f"\n--- Processing {noise_type} ---")
    
    # 1. Add noise
    print(f"Adding {noise_type}...")
    noisy_img = noise_func(image)
    
    # 2. Apply filters
    print("Applying Mean Filter...")
    mean_img = apply_mean_filter(noisy_img, kernel_size=5)
    
    print("Applying Median Filter...")
    median_img = apply_median_filter(noisy_img, kernel_size=5)
    
    print("Applying Gaussian Filter...")
    gaussian_img = apply_gaussian_filter(noisy_img, kernel_size=5, sigma=1.5)
    
    # 3. Calculate metrics
    metrics = {
        'Mean': {
            'MSE': calculate_mse(image, mean_img),
            'PSNR': calculate_psnr(image, mean_img)
        },
        'Median': {
            'MSE': calculate_mse(image, median_img),
            'PSNR': calculate_psnr(image, median_img)
        },
        'Gaussian': {
            'MSE': calculate_mse(image, gaussian_img),
            'PSNR': calculate_psnr(image, gaussian_img)
        }
    }
    
    print(f"Metrics for {noise_type}:")
    for f_name, mets in metrics.items():
        print(f"  {f_name} Filter -> MSE: {mets['MSE']:.2f}, PSNR: {mets['PSNR']:.2f} dB")
        
    # Find best filter based on PSNR
    best_filter = max(metrics, key=lambda k: metrics[k]['PSNR'])
    print(f"Recommended filter for {noise_type}: {best_filter}")
    
    # 4. Visualization
    plot_and_save_images(image, noisy_img, mean_img, median_img, gaussian_img, noise_type, output_dir)
    plot_metrics(metrics, noise_type, output_dir)
    
    # Save individual images for inspection
    noise_dir = os.path.join(output_dir, noise_type.replace(" ", "_"))
    os.makedirs(noise_dir, exist_ok=True)
    
    # Prefix output images with original filename
    base_name = image_name
    
    cv2.imwrite(os.path.join(noise_dir, f"{base_name}_1_original.jpg"), image)
    cv2.imwrite(os.path.join(noise_dir, f"{base_name}_2_noisy.jpg"), noisy_img)
    cv2.imwrite(os.path.join(noise_dir, f"{base_name}_3_mean.jpg"), mean_img)
    cv2.imwrite(os.path.join(noise_dir, f"{base_name}_4_median.jpg"), median_img)
    cv2.imwrite(os.path.join(noise_dir, f"{base_name}_5_gaussian.jpg"), gaussian_img)


def process_image(filepath, output_dir):
    image_name = os.path.splitext(os.path.basename(filepath))[0]
    # Read original image
    image = cv2.imread(filepath)
    if image is None:
        print(f"Error: Could not read image {filepath}.")
        return

    # Specific output dir for this image
    img_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(img_output_dir, exist_ok=True)
    
    print(f"\n{'='*40}")
    print(f"Processing Dataset Image: {image_name}")
    print(f"{'='*40}")
        
    # For Gaussian Noise
    process_noise_pipeline(
        image, 
        noise_type="Gaussian Noise", 
        noise_func=lambda img: add_gaussian_noise(img, var=0.02),
        output_dir=img_output_dir,
        image_name=image_name
    )
    
    # For Salt and Pepper Noise
    process_noise_pipeline(
        image, 
        noise_type="Salt and Pepper Noise", 
        noise_func=lambda img: add_salt_and_pepper_noise(img, amount=0.05),
        output_dir=img_output_dir,
        image_name=image_name
    )

def main():
    data_dir = "data"
    output_dir = "output"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    download_sample_images(data_dir)
    
    # Process all images in data_dir
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(data_dir, filename)
            process_image(filepath, output_dir)
    
    print(f"\nAll dataset processing complete. Check the '{output_dir}' directory for results.")

if __name__ == "__main__":
    main()
