import matplotlib.pyplot as plt
import cv2
import os

def plot_and_save_images(original, noisy, mean_filtered, median_filtered, gaussian_filtered, noise_type, output_dir):
    """Plots and saves the filter comparison images."""
    # Convert from BGR to RGB for matplotlib
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(bgr2rgb(original), cmap='gray' if len(original.shape)==2 else None)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(bgr2rgb(noisy), cmap='gray' if len(noisy.shape)==2 else None)
    plt.title(f'Noisy Image ({noise_type})')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(bgr2rgb(mean_filtered), cmap='gray' if len(mean_filtered.shape)==2 else None)
    plt.title('Mean Filter')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(bgr2rgb(median_filtered), cmap='gray' if len(median_filtered.shape)==2 else None)
    plt.title('Median Filter')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(bgr2rgb(gaussian_filtered), cmap='gray' if len(gaussian_filtered.shape)==2 else None)
    plt.title('Gaussian Filter')
    plt.axis('off')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'filter_comparison_{noise_type.replace(" ", "_")}.png'))
    plt.close()

def plot_metrics(metrics_dict, noise_type, output_dir):
    """Plots and saves MSE and PSNR metrics bar and line charts."""
    filters = list(metrics_dict.keys())
    mses = [metrics_dict[f]['MSE'] for f in filters]
    psnrs = [metrics_dict[f]['PSNR'] for f in filters]
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Filter Type')
    ax1.set_ylabel('MSE', color=color)
    ax1.bar([f + ' (MSE)' for f in filters], mses, color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('PSNR (dB)', color=color)  
    ax2.plot(filters, psnrs, color=color, marker='o', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  
    plt.title(f'Filter Performance on {noise_type}')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'metrics_{noise_type.replace(" ", "_")}.png'))
    plt.close()
