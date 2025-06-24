# color_image_processing_utils.py
import cv2
import numpy as np
import matplotlib.pyplot as plt # For example usage

def load_color_image(image_path):
    """Loads a color image using OpenCV."""
    color_image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color_image_bgr is None:
        print(f"Warning: Could not load color image from {image_path}")
    return color_image_bgr

def get_l_channel_from_lab(color_image_bgr):
    """Converts a BGR color image to Lab and returns the L* channel as uint8."""
    if color_image_bgr is None:
        return None
    try:
        lab_image = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2Lab)
        l_channel = lab_image[:,:,0]
        # Normalize L* channel to 0-255 uint8 if it's not already (L* is typically 0-100)
        l_channel_uint8 = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return l_channel_uint8
    except Exception as e:
        print(f"Error converting to Lab or extracting L* channel: {e}")
        return None

def get_v_channel_from_hsv(color_image_bgr):
    """Converts a BGR color image to HSV and returns the V (Value) channel as uint8."""
    if color_image_bgr is None:
        return None
    try:
        hsv_image = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv_image[:,:,2] # V channel is typically 0-255 uint8
        return v_channel 
    except Exception as e:
        print(f"Error converting to HSV or extracting V channel: {e}")
        return None

def get_s_channel_from_hsv(color_image_bgr):
    """Converts a BGR color image to HSV and returns the S (Saturation) channel as uint8."""
    if color_image_bgr is None:
        return None
    try:
        hsv_image = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2HSV)
        s_channel = hsv_image[:,:,1] # S channel is typically 0-255 uint8
        return s_channel
    except Exception as e:
        print(f"Error converting to HSV or extracting S channel: {e}")
        return None

if __name__ == '__main__':
    print("--- Testing Color Image Processing Utilities ---")
    # Create a dummy BGR image for testing
    dummy_color_bgr = np.random.randint(0, 256, (60, 80, 3), dtype=np.uint8)
    dummy_color_bgr[10:30, 10:30, :] = [255, 0, 0] # Blue patch
    dummy_color_bgr[30:50, 30:50, :] = [0, 255, 0] # Green patch
    
    print("Testing L* channel extraction from Lab...")
    l_channel = get_l_channel_from_lab(dummy_color_bgr)
    if l_channel is not None:
        print(f"L* channel shape: {l_channel.shape}, dtype: {l_channel.dtype}, min: {l_channel.min()}, max: {l_channel.max()}")
    
    print("\nTesting V channel extraction from HSV...")
    v_channel = get_v_channel_from_hsv(dummy_color_bgr)
    if v_channel is not None:
        print(f"V channel shape: {v_channel.shape}, dtype: {v_channel.dtype}, min: {v_channel.min()}, max: {v_channel.max()}")

    print("\nTesting S channel extraction from HSV...")
    s_channel = get_s_channel_from_hsv(dummy_color_bgr)
    if s_channel is not None:
        print(f"S channel shape: {s_channel.shape}, dtype: {s_channel.dtype}, min: {s_channel.min()}, max: {s_channel.max()}")

    # Example of displaying these channels
    if l_channel is not None and v_channel is not None and s_channel is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(cv2.cvtColor(dummy_color_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Dummy BGR")
        axes[0].axis('off')
        
        axes[1].imshow(l_channel, cmap='gray')
        axes[1].set_title("L* Channel (Lab)")
        axes[1].axis('off')

        axes[2].imshow(v_channel, cmap='gray')
        axes[2].set_title("V Channel (HSV)")
        axes[2].axis('off')
        
        axes[3].imshow(s_channel, cmap='gray')
        axes[3].set_title("S Channel (HSV)")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()