# fitness_otsu_variance.py
import numpy as np

def otsu_between_class_variance(hist, thresholds):
    """
    Calculates Otsu's between-class variance for given thresholds.
    The goal is to maximize this variance.
    Assumes hist is a normalized histogram of length 256.
    """
    thresholds_sorted = np.sort(thresholds)
    # Add boundaries 0 and 255 (max intensity level for 8-bit)
    full_thresholds = np.concatenate(([0], thresholds_sorted, [255])) 
    
    # Calculate total mean of the histogram (mu_T)
    # intensity_levels = np.arange(256)
    # total_mean = np.sum(intensity_levels * hist)
    # More robust calculation for total_mean if hist isn't perfectly 256 length (though it should be)
    total_mean = 0
    for i in range(len(hist)):
        total_mean += i * hist[i]

    between_class_variance = 0.0
    epsilon = 1e-12 # To avoid division by zero

    for i in range(len(full_thresholds) - 1):
        start = int(np.round(full_thresholds[i]))
        end = int(np.round(full_thresholds[i+1]))

        if start >= end: # Skip empty or invalid segment
            continue

        segment_hist_slice = hist[start:end]
        prob_mass = segment_hist_slice.sum() # w_k: probability of class k

        if prob_mass < epsilon: # Skip class with negligible probability mass
            continue

        # mu_k: mean of class k
        # Need to ensure intensity_values align with segment_hist_slice
        # class_intensity_levels = np.arange(start, end) # Levels belonging to this class
        # segment_mean_numerator = np.sum(class_intensity_levels * segment_hist_slice)
        
        # More robust calculation of segment mean numerator:
        segment_mean_numerator = 0
        for j, intensity_val in enumerate(range(start, end)):
            if j < len(segment_hist_slice): # Ensure we don't go out of bounds for segment_hist_slice
                 segment_mean_numerator += intensity_val * segment_hist_slice[j]
        
        segment_mean = segment_mean_numerator / prob_mass if prob_mass > epsilon else 0
            
        between_class_variance += prob_mass * ((segment_mean - total_mean) ** 2)
        
    if not np.isfinite(between_class_variance): 
        return -np.inf # Should be maximized, so -inf is bad for errors
    return between_class_variance

if __name__ == '__main__':
    # Example usage (for testing this module independently)
    print("--- Testing Otsu's Between-Class Variance Function ---")
    # Create a dummy histogram (e.g., bimodal)
    dummy_hist = np.zeros(256)
    dummy_hist[50:100] = 1.0 # First mode
    dummy_hist[150:200] = 0.8 # Second mode
    dummy_hist = dummy_hist / dummy_hist.sum() # Normalize

    test_thresholds_1 = [120] # One threshold
    variance1 = otsu_between_class_variance(dummy_hist, test_thresholds_1)
    print(f"Variance for thresholds {test_thresholds_1}: {variance1:.6f}")

    test_thresholds_2 = [80, 160] # Two thresholds
    variance2 = otsu_between_class_variance(dummy_hist, test_thresholds_2)
    print(f"Variance for thresholds {test_thresholds_2}: {variance2:.6f}")

    test_thresholds_bad = [100] # A potentially less optimal threshold
    variance_bad = otsu_between_class_variance(dummy_hist, test_thresholds_bad)
    print(f"Variance for thresholds {test_thresholds_bad}: {variance_bad:.6f} (expect lower for non-optimal)")