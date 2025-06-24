# segmentation_evaluation.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity # Renamed for clarity
import matplotlib.pyplot as plt # For example usage

def load_ground_truth_image(image_path, grayscale=True):
    """Loads a ground truth image, optionally converts to grayscale."""
    try:
        if grayscale:
            gt_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if gt_image is None:
            print(f"Warning: Could not load ground truth image from {image_path}")
        return gt_image
    except Exception as e:
        print(f"Error loading ground truth image {image_path}: {e}")
        return None

def calculate_ssim(image1, image2, data_range_val=None, win_size=7, return_diff_image=False):
    """
    Calculates Structural Similarity Index (SSIM) between two images.
    Ensure images are of the same shape and dtype (typically uint8 or float).
    data_range_val: The dynamic range of the images (e.g., 255 for uint8). If None, estimated from image1.
    win_size: The side-length of the sliding window used in comparison. Must be odd.
    return_diff_image: If True, also returns the difference image.
    """
    if image1 is None or image2 is None:
        print("Error: One or both images for SSIM are None.")
        return (0.0, None) if return_diff_image else 0.0
        
    if image1.shape != image2.shape:
        print(f"Error: Image shapes do not match for SSIM. Img1: {image1.shape}, Img2: {image2.shape}")
        return (0.0, None) if return_diff_image else 0.0

    if image1.ndim > 2 and image1.shape[2] > 1: # If multichannel (color)
        multichannel_flag = True
        # For scikit-image SSIM, channel axis should be the last one by default if not specified.
        # OpenCV BGR vs RGB: if consistency is needed, convert one. Assuming they are compatible for now.
    else:
        multichannel_flag = False

    if data_range_val is None:
        data_range_val = image1.max() - image1.min()
        if data_range_val == 0: # Handle flat image case
            # If both are flat and identical, SSIM is 1. If different, it's more complex.
            # For safety, if flat, assume a common range like 255 for uint8.
            # This might need problem-specific handling.
            data_range_val = 255 if image1.dtype == np.uint8 else 1.0


    # Ensure win_size is appropriate
    min_dim = min(image1.shape[:2])
    if win_size > min_dim:
        win_size = max(3, min_dim // 2 * 2 -1 ) # Make it odd and smaller
        if win_size < 3 : win_size = min(3, min_dim if min_dim % 2 != 0 else min_dim -1)
        if win_size <3: win_size = None # if image too small for default, skimage handles it.
        print(f"Adjusted SSIM win_size to {win_size} due to small image dimensions.")


    try:
        if return_diff_image:
            score, diff_img = structural_similarity(
                image1, image2, 
                data_range=data_range_val, 
                win_size=win_size, 
                full=True, 
                channel_axis=-1 if multichannel_flag else None,
                gaussian_weights=True # Often gives better results
            )
            # Diff image from SSIM is often (1 - ssim_map) / 2. Normalize for display if needed.
            if diff_img is not None:
                 diff_img = (diff_img * 255).astype(np.uint8) if diff_img.max() <=1.0 else diff_img.astype(np.uint8)

            return score, diff_img
        else:
            score = structural_similarity(
                image1, image2, 
                data_range=data_range_val, 
                win_size=win_size, 
                full=False, 
                channel_axis=-1 if multichannel_flag else None,
                gaussian_weights=True
            )
            return score
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return (0.0, None) if return_diff_image else 0.0


def explain_other_metrics():
    print("\n--- Other Potential Evaluation Metrics (Conceptual) ---")
    print("""
    Besides SSIM, other metrics are common for segmentation evaluation if ground truth is available:

    1.  Pixel Accuracy:
        - Formula: (Number of Correctly Classified Pixels) / (Total Number of Pixels)
        - Pros: Simple to calculate.
        - Cons: Can be misleading for imbalanced classes.

    2.  Intersection over Union (IoU) / Jaccard Index:
        - Formula (per class): (Area of Overlap between Prediction and GT) / (Area of Union)
        - Mean IoU (mIoU): Average IoU across all classes.
        - Pros: Robust metric, widely used in semantic segmentation. Penalizes false positives and false negatives.
        - Implementation: Requires comparing class labels. For each class, create binary masks for
          prediction and GT, then compute TP, FP, FN to get IoU = TP / (TP + FP + FN).

    3.  Dice Coefficient (F1 Score):
        - Formula (per class): 2 * (Area of Overlap) / (Sum of Pixels in Prediction + Sum of Pixels in GT)
        - Relation to IoU: Dice = 2 * IoU / (IoU + 1)
        - Pros: Similar to IoU, widely used.
        - Implementation: Similar to IoU, requires binary masks per class. Dice = 2*TP / (2*TP + FP + FN).

    These metrics typically require the segmented image and ground truth to have class labels
    (e.g., pixel values 0, 1, 2... for different segments) rather than continuous intensity values.
    The output of `apply_thresholds` (which assigns mean intensities) would need to be converted
    to class labels first (e.g., by mapping ranges to 0, 1, 2...).
    """)

if __name__ == '__main__':
    print("--- Testing Segmentation Evaluation Utilities ---")
    # Create dummy images for SSIM testing
    img_a = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    img_b = img_a.copy() 
    img_b[10:20, 10:20] = np.clip(img_b[10:20, 10:20] + 50, 0, 255) # Introduce some difference

    print("Calculating SSIM between two similar grayscale images:")
    ssim_score_val, diff_image = calculate_ssim(img_a, img_b, data_range_val=255, return_diff_image=True)
    print(f"SSIM Score: {ssim_score_val:.4f}")

    if diff_image is not None:
        fig_s, axes_s = plt.subplots(1, 3, figsize=(12, 4))
        axes_s[0].imshow(img_a, cmap='gray'); axes_s[0].set_title("Image A")
        axes_s[1].imshow(img_b, cmap='gray'); axes_s[1].set_title("Image B (Modified A)")
        axes_s[2].imshow(diff_image, cmap='viridis'); axes_s[2].set_title("SSIM Difference Map")
        for ax_s in axes_s: ax_s.axis('off')
        plt.show()

    # Test with dummy color images
    color_img_a = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    color_img_b = color_img_a.copy()
    color_img_b[10:20, 10:20, 0] = np.clip(color_img_b[10:20, 10:20, 0] + 50, 0, 255) # Diff in one channel

    print("\nCalculating SSIM between two similar color images:")
    ssim_score_color = calculate_ssim(color_img_a, color_img_b, data_range_val=255, win_size=7) # skimage handles multichannel if channel_axis is right
    print(f"SSIM Score (Color): {ssim_score_color:.4f}")


    explain_other_metrics()