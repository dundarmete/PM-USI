# main_pso_segmenter.py
# %% Import necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from skimage import data # <<< ADDED for astronaut image
import time
import os
# import imageio # Uncomment if using imageio for TIFF reading

# --- Potentially import functions from extension files ---
from fitness_otsu_variance import otsu_between_class_variance
from color_image_processing_utils import get_l_channel_from_lab, get_v_channel_from_hsv # load_color_image is now for specific demo
from segmentation_evaluation import calculate_ssim, load_ground_truth_image


# %% Helper Functions (Core to this main script)
# ... (calculate_histogram, kapur_entropy, apply_thresholds functions remain unchanged) ...
def calculate_histogram(image):
    """Calculates the normalized histogram of a grayscale image."""
    if image.dtype != np.uint8:
        if image.max() <= 1.0 and image.min() >=0.0:
             image = (image * 255).astype(np.uint8)
        else:
             image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum() # Normalize
    return hist

def kapur_entropy(hist, thresholds):
    """
    Calculates Kapur's entropy for a given histogram and thresholds.
    F(T) = sum H_i, where H_i = - sum_{j in class i} (p_j/w_i) * log(p_j/w_i)
    """
    thresholds_sorted = np.sort(thresholds)
    # Add boundaries 0 and 255 (max intensity level for 8-bit)
    full_thresholds = np.concatenate(([0], thresholds_sorted, [255]))
    total_entropy = 0.0
    epsilon = 1e-12 # Increased precision for log and division

    for i in range(len(full_thresholds) - 1):
        start = int(np.round(full_thresholds[i]))
        end = int(np.round(full_thresholds[i+1]))

        if start >= end: # Skip empty or invalid segment
            continue

        segment_hist_slice = hist[start:end] # Probabilities p_j for this segment
        prob_mass = segment_hist_slice.sum() # This is w_i (probability mass of the class)

        if prob_mass < epsilon:
            # Segment is empty or has negligible probability mass, H_i = 0
            continue

        # Calculate H_i = - sum ( (p_j/w_i) * log(p_j/w_i) )
        terms = segment_hist_slice[segment_hist_slice > epsilon] / prob_mass
        segment_class_entropy = -np.sum(terms * np.log(terms + epsilon))
        
        total_entropy += segment_class_entropy # Sum of H_i for each class

    if not np.isfinite(total_entropy): # Should not happen with epsilon handling
        return -np.inf # Fitness is maximized, so -inf is bad
    return total_entropy


def apply_thresholds(image, thresholds):
    """Applies thresholds to segment the image."""
    thresholds = np.sort(thresholds)
    segmented_image = np.zeros_like(image)
    img_min_val = 0 # Assuming 8-bit image range start
    img_max_val = 255 # Assuming 8-bit image range end
    boundaries = np.concatenate(([img_min_val], thresholds, [img_max_val]))

    for i in range(len(boundaries) - 1):
        lower_bound = boundaries[i]
        upper_bound = boundaries[i+1]
        # Assign a representative value for the segment, e.g., midpoint or a class label
        segment_value = int((lower_bound + upper_bound) / 2 ) # Example: midpoint intensity

        mask = (image >= lower_bound) & (image < upper_bound)
        if i == len(boundaries) - 2: # Last segment includes the upper_bound
            mask = (image >= lower_bound) & (image <= upper_bound)
        segmented_image[mask] = segment_value
    return segmented_image

# %% Particle Swarm Optimization (PSO) Implementation
# ... (Particle class and pso_segmentation function remain unchanged from your working version) ...
class Particle:
    """Represents a particle in the PSO swarm."""
    def __init__(self, num_thresholds, actual_image_intensity_range):
        self.num_thresholds = num_thresholds
        self.min_intensity, self.max_intensity = actual_image_intensity_range

        if self.min_intensity >= self.max_intensity:
            # print(f"Warning: Particle init with problematic range [{self.min_intensity}, {self.max_intensity}]. Adjusting.")
            if self.max_intensity > 1: self.min_intensity = max(1, self.max_intensity -1)
            else: self.min_intensity = 1; self.max_intensity = 2 # Default small range for safety
            if self.min_intensity >= self.max_intensity: self.min_intensity = 1; self.max_intensity = 254 # Fallback


        available_unique_values = self.max_intensity - self.min_intensity + 1
        if available_unique_values < num_thresholds and num_thresholds > 0 :
            # print(f"Warning: Not enough unique values ({available_unique_values}) in particle range [{self.min_intensity}, {self.max_intensity}] for {num_thresholds} thresholds. Using linspace.")
            self.position = np.linspace(self.min_intensity, self.max_intensity, num_thresholds, dtype=int)
            self.position = np.unique(self.position) # Take unique values
            if len(self.position) < num_thresholds: # Pad if some were lost
                needed = num_thresholds - len(self.position)
                padding_values = np.full(needed, self.position[-1] if len(self.position)>0 else self.min_intensity)
                self.position = np.sort(np.concatenate((self.position, padding_values)))
        elif num_thresholds == 0:
            self.position = np.array([])
        else:
            self.position = np.sort(np.random.randint(self.min_intensity, self.max_intensity + 1, num_thresholds))
            attempts = 0; max_attempts = 10 # Try to get unique thresholds
            while len(np.unique(self.position)) < num_thresholds and attempts < max_attempts:
                 self.position = np.sort(np.random.randint(self.min_intensity, self.max_intensity + 1, num_thresholds))
                 attempts += 1
            if len(np.unique(self.position)) < num_thresholds: # Fallback if still not unique
                self.position = np.unique(self.position)
                if len(self.position) < num_thresholds: # Pad if some were lost
                    needed = num_thresholds - len(self.position)
                    padding_values = np.random.randint(self.min_intensity, self.max_intensity + 1, needed)
                    self.position = np.sort(np.concatenate((self.position, padding_values)))


        v_range = (self.max_intensity - self.min_intensity) * 0.1 if self.max_intensity > self.min_intensity else 1.0
        self.velocity = np.random.uniform(-v_range, v_range, num_thresholds)

        self.pbest_position = self.position.copy()
        self.pbest_fitness = -np.inf
        self.current_fitness = -np.inf

    def update_velocity(self, gbest_position, w, c1, c2):
        if self.num_thresholds == 0: return # No thresholds, no velocity update
        r1 = np.random.rand(self.num_thresholds)
        r2 = np.random.rand(self.num_thresholds)
        self.velocity = (w * self.velocity +
                         c1 * r1 * (self.pbest_position - self.position) +
                         c2 * r2 * (gbest_position - self.position))
        # Limit velocity
        max_vel_val = (self.max_intensity - self.min_intensity) * 0.2 if self.max_intensity > self.min_intensity else 0.5
        self.velocity = np.clip(self.velocity, -max_vel_val, max_vel_val)


    def update_position(self):
        if self.num_thresholds == 0: return # No thresholds, no position update
        new_position = self.position + self.velocity
        new_position = np.clip(new_position, self.min_intensity, self.max_intensity)
        new_position = np.round(new_position).astype(int)
        new_position = np.sort(np.unique(new_position))

        # Ensure position maintains the correct number of thresholds
        if len(new_position) < self.num_thresholds:
            needed = self.num_thresholds - len(new_position)
            fill_values = []
            current_pos_set = set(new_position)
            for _ in range(needed): # Try to add distinct values
                added = False
                for attempt in range(5): 
                    candidate = np.random.randint(self.min_intensity, self.max_intensity + 1)
                    if candidate not in current_pos_set and candidate not in fill_values:
                        fill_values.append(candidate); added = True; break
                if not added: # Fallback if couldn't find distinct quickly
                    fill_values.append(np.random.randint(self.min_intensity, self.max_intensity + 1))
            
            new_position = np.sort(np.concatenate((new_position, fill_values)))
            new_position = np.unique(new_position) # Ensure uniqueness again
            # If still not enough (very unlikely with fallback), or too many (also unlikely)
            if len(new_position) > self.num_thresholds: new_position = new_position[:self.num_thresholds]
            elif len(new_position) < self.num_thresholds : # Final padding if absolutely necessary
                padding_needed = self.num_thresholds - len(new_position)
                # Pad with last value or a random value (less ideal, but ensures correct dimension)
                last_val = new_position[-1] if len(new_position) > 0 else self.min_intensity
                new_position = np.sort(np.concatenate((new_position, [last_val] * padding_needed)))
        self.position = new_position


    def evaluate_fitness(self, hist, fitness_func):
        if self.num_thresholds > 0 and len(self.position) != self.num_thresholds:
            # print(f"Warning: Particle position length {len(self.position)} != num_thresholds {self.num_thresholds}. Setting poor fitness.")
            self.current_fitness = -np.inf # Invalid state
        elif self.num_thresholds == 0: # No thresholds, segmentation is just one class (the whole image)
            # The fitness could be the entropy of the entire histogram (treated as one class)
            # For Kapur's, if thresholds list is empty, it implies one class [0, 255]
            self.current_fitness = fitness_func(hist, []) if fitness_func is kapur_entropy else 0.0
        else:
            self.current_fitness = fitness_func(hist, self.position)

        if self.current_fitness > self.pbest_fitness:
            self.pbest_fitness = self.current_fitness
            self.pbest_position = self.position.copy()


def pso_segmentation(image, num_thresholds, swarm_size=30, max_iter=100,
                     w_start=0.9, w_end=0.4, c1=1.5, c2=1.5,
                     refresh_ratio=0.1,
                     fitness_func=kapur_entropy, # Default fitness function
                     verbose=True):
    start_time = time.time()
    hist = calculate_histogram(image)

    # --- Define threshold search range based on actual image content ---
    img_actual_min_val = image.min()
    img_actual_max_val = image.max()

    particle_thr_min = img_actual_min_val + 1
    particle_thr_max = img_actual_max_val - 1
    particle_thr_min = max(1, particle_thr_min) # Global lower bound for thresholds
    particle_thr_max = min(254, particle_thr_max) # Global upper bound for thresholds
    
    if particle_thr_min >= particle_thr_max :
        # print(f"Warning: Calculated particle threshold range [{particle_thr_min}, {particle_thr_max}] is invalid or too narrow. Adjusting.")
        if img_actual_max_val > img_actual_min_val : # If there's any range in image
            particle_thr_min = img_actual_min_val # Fallback to image min
            particle_thr_max = img_actual_max_val # Fallback to image max
        else: # Flat image
            particle_thr_min = 1
            particle_thr_max = 128 # Arbitrary
        
        particle_thr_min = max(1, particle_thr_min) # Ensure at least 1
        particle_thr_max = min(254, particle_thr_max) # Ensure at most 254
        if particle_thr_min >= particle_thr_max: # If still invalid (e.g. image_max is 0 or 1)
            # Create a minimal valid range, ensuring it can support num_thresholds if >0
            particle_thr_max = particle_thr_min + max(1, num_thresholds if num_thresholds > 0 else 1)
            particle_thr_max = min(254, particle_thr_max)
            if particle_thr_min >= particle_thr_max: particle_thr_min = max(0, particle_thr_max -1) # Final safety


    actual_particle_search_range = (particle_thr_min, particle_thr_max)
    if verbose:
        print(f"Image actual content range: [{img_actual_min_val}, {img_actual_max_val}]")
        print(f"Particle thresholds will be sought in range: [{actual_particle_search_range[0]}, {actual_particle_search_range[1]}]")

    if num_thresholds == 0: # Handle case of 0 thresholds (entire image is one segment)
        if verbose: print("PSO with 0 thresholds. Image is one segment.")
        # Fitness for 0 thresholds is the entropy/variance of the whole image (one class)
        fitness_val = fitness_func(hist, []) if num_thresholds == 0 else 0.0
        if verbose: print(f"Fitness (0 thresholds): {fitness_val:.6f}")
        return np.array([]), [fitness_val] * max_iter # Return empty thresholds and dummy history


    # --- Initialize Swarm ---
    swarm = [Particle(num_thresholds, actual_particle_search_range) for _ in range(swarm_size)]

    gbest_position = None
    gbest_fitness = -np.inf
    
    # Initialize gbest from the initial swarm
    for p_init in swarm:
        p_init.evaluate_fitness(hist, fitness_func) # Evaluate initial positions
        if p_init.pbest_fitness > gbest_fitness:
            gbest_fitness = p_init.pbest_fitness
            gbest_position = p_init.pbest_position.copy()
    if gbest_position is None and swarm : # Fallback if all initial fitnesses were -inf (unlikely with proper init)
        # This might happen if num_thresholds is problematic for the range initially
        gbest_position = swarm[0].pbest_position.copy() # Take the first particle's pbest
        gbest_fitness = swarm[0].pbest_fitness # And its fitness
    
    gbest_fitness_history = []


    for iteration in range(max_iter):
        w = w_start - (w_start - w_end) * (iteration / max_iter)
        
        # current_iter_gbest_fitness_candidate = gbest_fitness # Start with overall gbest
        # current_iter_gbest_position_candidate = gbest_position # Not needed, direct update to gbest
        
        for particle in swarm:
            # Particle position should always have num_thresholds elements.
            if len(particle.position) == num_thresholds:
                 particle.evaluate_fitness(hist, fitness_func)
            else: # Should ideally not happen with robust particle position management
                 # print(f"Warning: Particle has {len(particle.position)} thresholds. Expected {num_thresholds}. Assigning poor fitness.")
                 particle.current_fitness = -np.inf
                 particle.pbest_fitness = -np.inf # Reset pbest as it might be based on invalid state

            # Update overall global best if this particle's pbest is better
            if particle.pbest_fitness > gbest_fitness:
                gbest_fitness = particle.pbest_fitness
                gbest_position = particle.pbest_position.copy()
        
        gbest_fitness_history.append(gbest_fitness if np.isfinite(gbest_fitness) else (gbest_fitness_history[-1] if gbest_fitness_history else -np.inf) )

        if verbose and (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1}/{max_iter}, Best Fitness: {gbest_fitness:.6f}")

        # Particle Refresh Strategy
        if refresh_ratio > 0 and iteration < max_iter -1 : # Don't refresh on the very last iteration
            num_to_refresh = int(swarm_size * refresh_ratio)
            if num_to_refresh > 0:
                # Sort particles by their current fitness (ascending, so worst are first)
                particle_fitness_tuples = sorted([(p.current_fitness if np.isfinite(p.current_fitness) else -1e18, i) for i, p in enumerate(swarm)], key=lambda x: x[0])
                for k in range(num_to_refresh):
                    worst_particle_index = particle_fitness_tuples[k][1]
                    swarm[worst_particle_index] = Particle(num_thresholds, actual_particle_search_range)
                    # Newly refreshed particles will be evaluated at the start of the next iteration
        
        # Update velocities and positions
        if gbest_position is not None and len(gbest_position) == num_thresholds : # Ensure gbest is valid before using
            for particle in swarm:
                if len(particle.position) == num_thresholds: # Ensure particle state is valid
                     particle.update_velocity(gbest_position, w, c1, c2)
                     particle.update_position()
        elif gbest_position is None and iteration == 0 and swarm: 
             # This case should ideally be covered by initial gbest setup.
             # If still None, it indicates a deeper issue with initialization or fitness.
             pass


    end_time = time.time()
    if verbose:
        print(f"\nPSO finished in {end_time - start_time:.2f} seconds.")
        if gbest_position is not None:
            print(f"Best thresholds found: {np.round(gbest_position).astype(int)}")
        else:
            print("No valid global best position found.")
        print(f"Best Fitness ({fitness_func.__name__}): {gbest_fitness:.6f}")

    if gbest_position is not None and len(gbest_position) == num_thresholds:
        return np.round(gbest_position).astype(int), gbest_fitness_history
    else: # Fallback if no valid gbest was found
        # print("Warning: PSO did not find a valid set of thresholds. Returning fallback.")
        if img_actual_max_val > img_actual_min_val and num_thresholds > 0:
            # Create evenly spaced thresholds within the image's actual content range
            fallback_thresholds = np.linspace(img_actual_min_val +1, img_actual_max_val-1, num_thresholds, dtype=int)
            fallback_thresholds = np.unique(fallback_thresholds) # Ensure unique
            if len(fallback_thresholds) < num_thresholds: # Pad if too few due to narrow range/rounding
                padding = np.linspace(fallback_thresholds[-1]+1 if fallback_thresholds.size > 0 else img_actual_min_val+1, 
                                      img_actual_max_val-1, 
                                      num_thresholds - len(fallback_thresholds), dtype=int)
                fallback_thresholds = np.sort(np.concatenate((fallback_thresholds, padding)))
            return fallback_thresholds[:num_thresholds], gbest_fitness_history # Ensure correct number
        else: # Absolute fallback for flat images or problematic scenarios
            return np.array([64, 128, 192][:num_thresholds] if num_thresholds > 0 else []), gbest_fitness_history

# %% Main Execution Block
if __name__ == "__main__":
    # --- Parameters for MAIN PSO run (e.g., on your local MRI) ---
    # This part remains for your primary experiment with the local MRI tiff
    MAIN_IMAGE_PATH = '/Users/metehandundar/Desktop/Particle Methods/HW5/inputs/mri-stack.tif' # YOUR MRI STACK
    MAIN_SLICE_INDEX = 13
    IS_MAIN_IMAGE_COLOR = False # Your MRI stack slice is processed as grayscale L* equivalent

    NUM_THRESHOLDS = 3
    SWARM_SIZE = 40
    MAX_ITERATIONS = 100
    PSO_C1 = 1.5
    PSO_C2 = 1.5
    PSO_W_START = 0.9
    PSO_W_END = 0.4
    REFRESH_RATIO = 0.10

    # --- Load MAIN image for PSO (Your MRI Slice) ---
    if not os.path.exists(MAIN_IMAGE_PATH):
        print(f"Error: Main image file not found at {MAIN_IMAGE_PATH}")
        exit()

    image_to_process_for_pso = None
    display_image_original = None # This will hold the image to display as "original"

    if IS_MAIN_IMAGE_COLOR: # If your main image path was a color image
        # This block would load a color image and extract L* for PSO
        # For your current setup, this block is effectively skipped as IS_MAIN_IMAGE_COLOR = False
        print(f"--- Processing Main Image as COLOR: {MAIN_IMAGE_PATH} ---")
        main_loaded_color_image = cv2.imread(MAIN_IMAGE_PATH, cv2.IMREAD_COLOR)
        if main_loaded_color_image is not None:
            image_to_process_for_pso = get_l_channel_from_lab(main_loaded_color_image)
            display_image_original = cv2.cvtColor(main_loaded_color_image, cv2.COLOR_BGR2RGB)
        else:
            print(f"Failed to load main color image from {MAIN_IMAGE_PATH}. Exiting.")
            exit()
    else: # Grayscale TIFF stack processing (Your current primary workflow)
        print(f"--- Processing Main Image as Grayscale TIFF: {MAIN_IMAGE_PATH}, Slice: {MAIN_SLICE_INDEX} ---")
        try:
            retval, mats = cv2.imreadmulti(MAIN_IMAGE_PATH, [], cv2.IMREAD_UNCHANGED)
            if not retval or not mats:
                import imageio # Attempt fallback
                image_stack = imageio.vimread(MAIN_IMAGE_PATH)
                if MAIN_SLICE_INDEX >= len(image_stack): MAIN_SLICE_INDEX = min(len(image_stack) - 1, len(image_stack) // 2)
                image_slice = image_stack[MAIN_SLICE_INDEX]
            else:
                if MAIN_SLICE_INDEX >= len(mats): MAIN_SLICE_INDEX = min(len(mats) - 1, len(mats) // 2)
                image_slice = mats[MAIN_SLICE_INDEX]

            if image_slice is None: print(f"Error: Could not extract slice {MAIN_SLICE_INDEX}."); exit()
            if image_slice.ndim == 3 and image_slice.shape[-1] == 1: image_slice = image_slice.squeeze(axis=-1)
            if image_slice.ndim != 2: print(f"Error: Selected slice has {image_slice.ndim} dimensions."); exit()
            if image_slice.dtype == np.uint8 and image_slice.max() <= 255 and image_slice.min() >=0: image_to_process_for_pso = image_slice
            else:
                if image_slice.min() == image_slice.max(): image_to_process_for_pso = np.full_like(image_slice, 128, dtype=np.uint8)
                else: image_to_process_for_pso = cv2.normalize(image_slice, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            display_image_original = image_to_process_for_pso # For grayscale, original processed is for display
        except Exception as e: print(f"Error during main image loading: {e}"); exit()

    if image_to_process_for_pso is None: print("Failed to load and prepare main image for PSO."); exit()
        
    # --- Run MAIN PSO Segmentation (Default: Kapur's Entropy) ---
    print("\n--- Running MAIN PSO with Kapur's Entropy (on MRI L* or Grayscale) ---")
    current_fitness_function = kapur_entropy
    best_thresholds_pso, fitness_history = pso_segmentation(
        image_to_process_for_pso, num_thresholds=NUM_THRESHOLDS, swarm_size=SWARM_SIZE, max_iter=MAX_ITERATIONS,
        w_start=PSO_W_START, w_end=PSO_W_END, c1=PSO_C1, c2=PSO_C2,
        refresh_ratio=REFRESH_RATIO, fitness_func=current_fitness_function, verbose=True
    )
    segmented_image_pso = apply_thresholds(image_to_process_for_pso, best_thresholds_pso)

    # --- Compare MAIN run with Skimage Otsu's Multi-Thresholding ---
    print(f"\n--- Skimage Otsu's Multi-Thresholding on MAIN image ({NUM_THRESHOLDS+1} classes) ---")
    otsu_available = False; otsu_thresholds_skimage = []
    segmented_image_otsu_skimage = np.zeros_like(image_to_process_for_pso)
    if NUM_THRESHOLDS >= 1:
        try:
            otsu_thresholds_skimage = threshold_multiotsu(image_to_process_for_pso, classes=NUM_THRESHOLDS + 1)
            segmented_image_otsu_skimage = apply_thresholds(image_to_process_for_pso, otsu_thresholds_skimage)
            otsu_fitness_eval = current_fitness_function(calculate_histogram(image_to_process_for_pso), otsu_thresholds_skimage)
            print(f"Skimage Otsu thresholds: {np.round(otsu_thresholds_skimage).astype(int)}")
            print(f"Skimage Otsu Fitness ({current_fitness_function.__name__}): {otsu_fitness_eval:.6f}")
            otsu_available = True
        except Exception as e: print(f"Could not run/evaluate Skimage Otsu's method: {e}")
    else: print("Otsu's (skimage) requires at least 1 threshold. Skipping.")

    # --- Display Main Results ---
    try: plt.style.use('seaborn-v0_8-darkgrid')
    except: plt.style.use('default')
    # (Plotting logic from your previous version, slightly adapted for clarity)
    fig_main, axes_main = plt.subplots(2, 2, figsize=(12, 10))
    fig_main_title = f"Main Segmentation: {os.path.basename(MAIN_IMAGE_PATH)}"
    if not IS_MAIN_IMAGE_COLOR: fig_main_title += f" (Slice {MAIN_SLICE_INDEX})"
    fig_main_title += f" - {NUM_THRESHOLDS} thresholds"
    fig_main.suptitle(fig_main_title, fontsize=14)
    axes_main = axes_main.ravel()
    ax_idx = 0
    # Plot 1: Original Image for PSO
    im_orig_main = axes_main[ax_idx].imshow(display_image_original, cmap='gray' if display_image_original.ndim==2 else None)
    axes_main[ax_idx].set_title('Original Image for PSO')
    axes_main[ax_idx].axis('off')
    if display_image_original.ndim==2: fig_main.colorbar(im_orig_main, ax=axes_main[ax_idx], fraction=0.046, pad=0.04)
    ax_idx +=1
    # Plot 2: PSO Segmented Image
    im_pso_main = axes_main[ax_idx].imshow(segmented_image_pso, cmap='nipy_spectral', vmin=0, vmax=255)
    axes_main[ax_idx].set_title(f'PSO ({current_fitness_function.__name__})\nThr: {best_thresholds_pso}')
    axes_main[ax_idx].axis('off')
    fig_main.colorbar(im_pso_main, ax=axes_main[ax_idx], fraction=0.046, pad=0.04)
    ax_idx +=1
    # Plot 3: PSO Convergence Graph
    axes_main[ax_idx].plot(range(1, len(fitness_history) + 1), fitness_history, marker='.', linestyle='-', color='b')
    axes_main[ax_idx].set_title(f"PSO Convergence ({current_fitness_function.__name__})")
    axes_main[ax_idx].set_xlabel("Iteration"); axes_main[ax_idx].set_ylabel("Best Fitness")
    axes_main[ax_idx].grid(True); ax_idx +=1
    # Plot 4: Skimage Otsu's Segmented Image
    if otsu_available:
        im_otsu_sk_main = axes_main[ax_idx].imshow(segmented_image_otsu_skimage, cmap='nipy_spectral', vmin=0, vmax=255)
        axes_main[ax_idx].set_title(f"Skimage Otsu's\nThr: {np.round(otsu_thresholds_skimage).astype(int)}")
        fig_main.colorbar(im_otsu_sk_main, ax=axes_main[ax_idx], fraction=0.046, pad=0.04)
    else: axes_main[ax_idx].text(0.5, 0.5, "Otsu (skimage) N/A", ha='center', va='center')
    axes_main[ax_idx].axis('off'); ax_idx +=1
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()


    # --- Demonstrating Extension Modules ---
    
    # Extension 1: Using Otsu's Between-Class Variance as Fitness
    run_pso_with_otsu_variance_demo = True # Set True to run this demo
    if run_pso_with_otsu_variance_demo:
        print("\n--- DEMO: Running PSO with Otsu's Between-Class Variance (on main processed image) ---")
        best_thresholds_pso_otsu_ext, fitness_history_otsu_ext = pso_segmentation(
            image_to_process_for_pso, num_thresholds=NUM_THRESHOLDS, fitness_func=otsu_between_class_variance, 
            swarm_size=SWARM_SIZE, max_iter=MAX_ITERATIONS, verbose=True # Pass other params as needed
        )
        segmented_image_pso_otsu_ext = apply_thresholds(image_to_process_for_pso, best_thresholds_pso_otsu_ext)
        
        plt.figure(figsize=(6,5))
        im_pso_otsu_demo = plt.imshow(segmented_image_pso_otsu_ext, cmap='nipy_spectral', vmin=0, vmax=255)
        plt.title(f"PSO (Otsu's Variance Demo) - Thr: {best_thresholds_pso_otsu_ext}\nFitness: {fitness_history_otsu_ext[-1]:.2f}")
        plt.axis('off'); plt.colorbar(im_pso_otsu_demo); plt.show()

    # Extension 2: Color Image Processing Demo (using skimage.data.astronaut())
    run_color_processing_demo = True # Set True to run this demo
    if run_color_processing_demo:
        print("\n--- DEMO: Color Image Processing with skimage.data.astronaut() ---")
        try:
            astro_img_rgb = data.astronaut() # skimage loads as RGB (H, W, 3)
            astro_img_bgr = cv2.cvtColor(astro_img_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV utils

            l_channel_astro = get_l_channel_from_lab(astro_img_bgr)
            v_channel_astro = get_v_channel_from_hsv(astro_img_bgr)

            fig_astro, axes_astro = plt.subplots(1, 3, figsize=(15, 5))
            axes_astro[0].imshow(astro_img_rgb); axes_astro[0].set_title("Astronaut (Original RGB)"); axes_astro[0].axis('off')
            if l_channel_astro is not None:
                axes_astro[1].imshow(l_channel_astro, cmap='gray'); axes_astro[1].set_title("L* Channel (Lab)"); axes_astro[1].axis('off')
            else: axes_astro[1].text(0.5,0.5, "L* N/A", ha='center', va='center'); axes_astro[1].axis('off')
            if v_channel_astro is not None:
                axes_astro[2].imshow(v_channel_astro, cmap='gray'); axes_astro[2].set_title("V Channel (HSV)"); axes_astro[2].axis('off')
            else: axes_astro[2].text(0.5,0.5, "V N/A", ha='center', va='center'); axes_astro[2].axis('off')
            plt.suptitle("Color Image Processing Demo (Astronaut)")
            plt.tight_layout(); plt.show()
            
            # Optionally, run PSO on one of these extracted channels from astronaut
            run_pso_on_astronaut_l_channel = True # Set True to also segment astronaut L*
            if run_pso_on_astronaut_l_channel and l_channel_astro is not None:
                print("\n--- Running PSO on Astronaut L* channel (example) ---")
                best_thresholds_astro_l, hist_astro_l = pso_segmentation(
                   l_channel_astro, num_thresholds=NUM_THRESHOLDS, fitness_func=kapur_entropy, 
                   swarm_size=SWARM_SIZE, max_iter=MAX_ITERATIONS, verbose=False # Reduced verbosity for demo
                )
                print(f"PSO on Astronaut L* done. Thresholds: {best_thresholds_astro_l}, Fitness: {hist_astro_l[-1]:.2f}")
                segmented_astro_l = apply_thresholds(l_channel_astro, best_thresholds_astro_l)
                plt.figure(figsize=(6,5))
                im_astro_seg = plt.imshow(segmented_astro_l, cmap='nipy_spectral', vmin=0, vmax=255); 
                plt.title(f"Segmented Astronaut L* (Kapur's)\nThr: {best_thresholds_astro_l}")
                plt.axis('off'); plt.colorbar(im_astro_seg); plt.show()
        except NameError:
            print("Skipping Astronaut demo as 'skimage.data' might not be available or correctly imported.")
        except Exception as e_astro:
            print(f"Error during Astronaut demo: {e_astro}")


    # Extension 3: Evaluation with Ground Truth (SSIM)
    run_ssim_evaluation_demo = True # Set True to run this demo
    
    # USER: IMPORTANT! Provide the paths to YOUR two specific PNG files for this SSIM demo.
    # 1. PATH_TO_YOUR_SEGMENTED_MRI_PNG: The MRI image that has already been segmented (e.g., by a previous PSO run or another method).
    # 2. PATH_TO_ITS_CORRESPONDING_GROUND_TRUTH_PNG: The ground truth segmentation for the image above.
    PATH_TO_YOUR_SEGMENTED_MRI_PNG = "/Users/metehandundar/Desktop/Particle Methods/HW5/inputs/my_segmented_mri_example.png"  # <<< USER MUST PROVIDE THIS PATH
    PATH_TO_ITS_CORRESPONDING_GROUND_TRUTH_PNG = "/Users/metehandundar/Desktop/Particle Methods/HW5/inputs/gt_for_my_segmented_mri_example.png" # <<< USER MUST PROVIDE THIS PATH

    if run_ssim_evaluation_demo:
        print("\n--- DEMO: Evaluating a specific Segmentation with SSIM using provided PNG files ---")
        
        if os.path.exists(PATH_TO_YOUR_SEGMENTED_MRI_PNG) and os.path.exists(PATH_TO_ITS_CORRESPONDING_GROUND_TRUTH_PNG):
            # Load your specific segmented image (as grayscale)
            my_specific_segmented_img = cv2.imread(PATH_TO_YOUR_SEGMENTED_MRI_PNG, cv2.IMREAD_GRAYSCALE)
            # Load its corresponding ground truth image (as grayscale)
            # The load_ground_truth_image function can be used here as well.
            its_corresponding_gt_img = load_ground_truth_image(PATH_TO_ITS_CORRESPONDING_GROUND_TRUTH_PNG, grayscale=True)
            
            if my_specific_segmented_img is not None and its_corresponding_gt_img is not None:
                print(f"Loaded for SSIM Demo: Segmented Image ({os.path.basename(PATH_TO_YOUR_SEGMENTED_MRI_PNG)}) and its GT ({os.path.basename(PATH_TO_ITS_CORRESPONDING_GROUND_TRUTH_PNG)})")
                try:
                    # Ensure GT is uint8 and normalized if it's not (e.g. if it was labels 0,1,2)
                    # For SSIM, inputs should ideally be in a similar representation.
                    if its_corresponding_gt_img.dtype != np.uint8 or its_corresponding_gt_img.max() > 255 or its_corresponding_gt_img.min() < 0:
                         gt_img_for_ssim_norm = cv2.normalize(its_corresponding_gt_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    else:
                         gt_img_for_ssim_norm = its_corresponding_gt_img
                    
                    # Ensure your specific segmented image is also uint8 (it should be if loaded correctly)
                    if my_specific_segmented_img.dtype != np.uint8:
                        my_specific_segmented_img_norm = cv2.normalize(my_specific_segmented_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    else:
                        my_specific_segmented_img_norm = my_specific_segmented_img


                    if my_specific_segmented_img_norm.shape == gt_img_for_ssim_norm.shape:
                        ssim_score_val, ssim_diff_image = calculate_ssim(
                            my_specific_segmented_img_norm, 
                            gt_img_for_ssim_norm,
                            data_range_val=255, # Assuming uint8 images after normalization
                            return_diff_image=True
                        )
                        print(f"SSIM score between your specific segmented MRI and its Ground Truth: {ssim_score_val:.4f}")

                        fig_ssim_demo, axes_ssim_demo = plt.subplots(1, 3, figsize=(15, 5))
                        axes_ssim_demo[0].imshow(my_specific_segmented_img_norm, cmap='nipy_spectral', vmin=0, vmax=255); axes_ssim_demo[0].set_title("Your Segmented MRI (Demo)"); axes_ssim_demo[0].axis('off')
                        axes_ssim_demo[1].imshow(gt_img_for_ssim_norm, cmap='nipy_spectral', vmin=0, vmax=255); axes_ssim_demo[1].set_title("Its Ground Truth (Demo)"); axes_ssim_demo[1].axis('off')
                        if ssim_diff_image is not None:
                            im_ssim_diff_demo = axes_ssim_demo[2].imshow(ssim_diff_image, cmap='viridis'); axes_ssim_demo[2].set_title(f"SSIM Difference Map (Score: {ssim_score_val:.3f})"); axes_ssim_demo[2].axis('off')
                            fig_ssim_demo.colorbar(im_ssim_diff_demo, ax=axes_ssim_demo[2], fraction=0.046, pad=0.04)
                        else:
                             axes_ssim_demo[2].text(0.5,0.5, "SSIM Diff N/A", ha='center', va='center'); axes_ssim_demo[2].axis('off')
                        plt.suptitle("SSIM Evaluation (Specific Demo PNGs)")
                        plt.show()
                    else:
                        print(f"Shape mismatch for SSIM Demo: Your Seg PNG {my_specific_segmented_img_norm.shape}, Its GT PNG {gt_img_for_ssim_norm.shape}")
                except Exception as e_ssim:
                    print(f"Error during SSIM calculation for demo PNGs: {e_ssim}")
            else:
                if my_specific_segmented_img is None: print(f"Could not load your segmented image PNG from {PATH_TO_YOUR_SEGMENTED_MRI_PNG}.")
                if its_corresponding_gt_img is None: print(f"Could not load its ground truth PNG from {PATH_TO_ITS_CORRESPONDING_GROUND_TRUTH_PNG}.")
        else:
            if not os.path.exists(PATH_TO_YOUR_SEGMENTED_MRI_PNG): print(f"Your segmented image PNG for SSIM demo not found: {PATH_TO_YOUR_SEGMENTED_MRI_PNG}")
            if not os.path.exists(PATH_TO_ITS_CORRESPONDING_GROUND_TRUTH_PNG): print(f"Its ground truth PNG for SSIM demo not found: {PATH_TO_ITS_CORRESPONDING_GROUND_TRUTH_PNG}")
            print("To run this SSIM demo, please provide valid paths to your two PNG files.")