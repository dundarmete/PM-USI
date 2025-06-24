import numpy as np
import time
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation # Added for video output
from functools import partial # Added for animation function

# ==============================================================================
# Configuration Parameters
# ==============================================================================

# --- General Simulation Parameters ---
L = 15.0
rc = 1.0
rc_sq = rc**2
m = 1.0
gamma = 4.5
sigma = 1.0
kBT_target = sigma**2 / (2 * gamma)
dt_default = 0.01
wall_width = rc

# --- Part a: Fluid Test Parameters ---
rho_a = 4.0
N_a = int(rho_a * L**2)
a_FF_a = 25.0

# --- Part b: Couette Flow Parameters ---
rho_b = 4.0
N_b = int(rho_b * L**2)
num_chains_b = 42
chain_structure_b = ['A', 'A', 'B', 'B', 'B', 'B', 'B']
num_particles_per_chain_b = len(chain_structure_b)
v_wall_b = 5.0
Ks_b = 100.0
rs_b = 0.1
def get_a_ij_b(type1, type2):
    a_ij_b_matrix = {
        ('A', 'A'): 50, ('A', 'B'): 25, ('A', 'F'): 25, ('A', 'W'): 200,
        ('B', 'B'): 1,  ('B', 'F'): 300, ('B', 'W'): 200,
        ('F', 'F'): 25, ('F', 'W'): 200,
        ('W', 'W'): 0
    }
    key = tuple(sorted((type1, type2)))
    return a_ij_b_matrix.get(key, 25.0)

# --- Part c: Poiseuille Flow Parameters ---
rho_c = 4.0
N_c = int(rho_c * L**2)
num_rings_c = 10
ring_size_c = 9
v_wall_c = 0.0
Ks_c = 100.0
rs_c = 0.3
F_body_c = np.array([0.3, 0.0])
def get_a_ij_c(type1, type2):
    a_ij_c_matrix = {
        ('A', 'A'): 50, ('A', 'F'): 25, ('A', 'W'): 200,
        ('F', 'F'): 25, ('F', 'W'): 200,
        ('W', 'W'): 0
    }
    key = tuple(sorted((type1, type2)))
    return a_ij_c_matrix.get(key, 25.0)

# Type mapping for colors
type_colors = {'F': 'blue', 'W': 'grey', 'A': 'red', 'B': 'green'}
default_color = 'black'

# Video parameters
VIDEO_FRAME_INTERVAL = 50 # Store position data every X steps for video frame
VIDEO_FPS = 15 # Frames per second for output video

# ==============================================================================
# Helper Functions
# ==============================================================================
def apply_pbc(pos, L):
    return pos % L

def minimum_image_distance(dr, L):
    return dr - L * np.rint(dr / L)

# ==============================================================================
# Cell List Implementation
# ==============================================================================
def build_cell_list(pos, L, rc):
    n_cells_dim = int(L / rc)
    if n_cells_dim == 0: n_cells_dim = 1
    cell_len = L / n_cells_dim
    cell_indices = (pos // cell_len).astype(int)
    cell_indices = np.clip(cell_indices, 0, n_cells_dim - 1)
    cells = defaultdict(list)
    for i in range(pos.shape[0]):
        cells[tuple(cell_indices[i])].append(i)
    return cells, n_cells_dim, cell_len

# ==============================================================================
# Force Calculations
# ==============================================================================
def calculate_dpd_forces(pos, vel, types, L, rc, rc_sq, gamma, sigma, kBT_target, get_a_ij_func, dt):
    N = pos.shape[0]
    forces = np.zeros((N, 2))
    rand_nums = {}
    cells, n_cells_dim, cell_len = build_cell_list(pos, L, rc)
    processed_pairs = set()

    for cell_idx_tuple in cells:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell_x = (cell_idx_tuple[0] + dx) % n_cells_dim
                neighbor_cell_y = (cell_idx_tuple[1] + dy) % n_cells_dim
                neighbor_cell_tuple = (neighbor_cell_x, neighbor_cell_y)

                if neighbor_cell_tuple in cells:
                    particles_in_cell = cells[cell_idx_tuple]
                    particles_in_neighbor = cells[neighbor_cell_tuple]

                    for i in particles_in_cell:
                        for j in particles_in_neighbor:
                            if i >= j: continue
                            pair_key = (i, j)
                            if pair_key in processed_pairs: continue

                            my_pos, other_pos = pos[i], pos[j]
                            my_vel, other_vel = vel[i], vel[j]
                            my_type, other_type = types[i], types[j]

                            dr_vec = minimum_image_distance(my_pos - other_pos, L)
                            r_ij_sq = np.dot(dr_vec, dr_vec)

                            if r_ij_sq < rc_sq:
                                r_ij = np.sqrt(r_ij_sq)
                                r_hat = dr_vec / r_ij
                                v_ij = my_vel - other_vel
                                r_hat_dot_v_ij = np.dot(r_hat, v_ij)
                                wR = 1.0 - r_ij / rc
                                wD = wR**2
                                a_ij = get_a_ij_func(my_type, other_type)
                                F_C = (a_ij * wR) * r_hat
                                F_D = (-gamma * wD * r_hat_dot_v_ij) * r_hat
                                if pair_key not in rand_nums:
                                    rand_nums[pair_key] = np.random.randn()
                                F_R = (sigma * wR * rand_nums[pair_key]) * r_hat
                                F_ij = F_C + F_D + F_R
                                forces[i] += F_ij
                                forces[j] -= F_ij
                                processed_pairs.add(pair_key)
    return forces, rand_nums

def calculate_bond_forces(pos, bonds, Ks, rs, L):
    N = pos.shape[0]
    bond_forces = np.zeros((N, 2))
    for i, j in bonds:
        dr_vec = minimum_image_distance(pos[i] - pos[j], L)
        r_ij = np.linalg.norm(dr_vec)
        if r_ij > 1e-9:
            r_hat = dr_vec / r_ij
            F_S = (Ks * (1.0 - r_ij / rs)) * r_hat
            bond_forces[i] += F_S
            bond_forces[j] -= F_S
    return bond_forces

# ==============================================================================
# Integrator
# ==============================================================================
def velocity_verlet_step(pos, vel, forces, dt, m, L, fixed_indices=None, wall_vel=None):
    N = pos.shape[0]
    fixed_mask = np.zeros(N, dtype=bool)
    wall_mask = np.zeros(N, dtype=bool)
    if fixed_indices is not None and len(fixed_indices) > 0: fixed_mask[fixed_indices] = True
    if wall_vel is not None:
        wall_indices = list(wall_vel.keys())
        if len(wall_indices) > 0: wall_mask[wall_indices] = True
    mobile_mask = ~fixed_mask & ~wall_mask
    mobile_indices = np.where(mobile_mask)[0]

    pos[mobile_indices] += vel[mobile_indices] * dt + 0.5 * (forces[mobile_indices] / m) * dt**2
    if wall_vel is not None:
        for idx, v_wall in wall_vel.items():
             if idx < N: pos[idx] += v_wall * dt
    pos = apply_pbc(pos, L)

    vel_intermediate = np.copy(vel)
    vel_intermediate[mobile_indices] += 0.5 * (forces[mobile_indices] / m) * dt
    if fixed_indices is not None: vel_intermediate[fixed_indices] = 0.0
    if wall_vel is not None:
        for idx, v_wall in wall_vel.items():
            if idx < N: vel_intermediate[idx] = v_wall
    return pos, vel_intermediate

def velocity_update(vel, vel_intermediate, forces_new, dt, m, fixed_indices=None, wall_vel=None):
    N = vel.shape[0]
    vel_new = np.copy(vel_intermediate)
    fixed_mask = np.zeros(N, dtype=bool)
    wall_mask = np.zeros(N, dtype=bool)
    if fixed_indices is not None and len(fixed_indices) > 0: fixed_mask[fixed_indices] = True
    if wall_vel is not None:
        wall_indices = list(wall_vel.keys())
        if len(wall_indices) > 0: wall_mask[wall_indices] = True
    mobile_mask = ~fixed_mask & ~wall_mask
    mobile_indices = np.where(mobile_mask)[0]

    vel_new[mobile_indices] += 0.5 * (forces_new[mobile_indices] / m) * dt
    if fixed_indices is not None: vel_new[fixed_indices] = 0.0
    if wall_vel is not None:
        for idx, v_wall in wall_vel.items():
            if idx < N: vel_new[idx] = v_wall
    return vel_new

# ==============================================================================
# Initialization Functions
# ==============================================================================
def init_part_a(N, L, dt):
    """Initializes system for Part a: Fluid only."""
    print(f"Initializing Part a: N={N}, L={L}, dt={dt}")
    pos = np.random.rand(N, 2) * L
    vel = np.zeros((N, 2))
    types = np.array(['F'] * N)
    bonds = []
    fixed_indices = None
    wall_vel = None
    get_a_ij_func = lambda t1, t2: a_FF_a
    body_force = None
    molecule_indices = None # No molecules in part a
    return pos, vel, types, bonds, fixed_indices, wall_vel, get_a_ij_func, body_force, molecule_indices

def init_part_b(N, L, dt):
    """Initializes system for Part b: Couette flow with chains."""
    print(f"Initializing Part b: N={N}, L={L}, dt={dt}")
    n_mol_particles = num_chains_b * num_particles_per_chain_b
    if n_mol_particles > N: raise ValueError(f"Molecule particles ({n_mol_particles}) > N ({N})")

    types = []; bonds = []; molecule_indices = []
    start_idx = 0
    for _ in range(num_chains_b):
        chain_indices = list(range(start_idx, start_idx + num_particles_per_chain_b))
        molecule_indices.extend(chain_indices)
        types.extend(chain_structure_b)
        for i in range(num_particles_per_chain_b - 1):
            bonds.append(tuple(sorted((chain_indices[i], chain_indices[i+1]))))
        start_idx += num_particles_per_chain_b

    n_fluid = N - n_mol_particles
    types.extend(['F'] * n_fluid); types = np.array(types)
    pos = np.random.rand(N, 2) * L; vel = np.zeros((N, 2))
    fixed_indices = None; wall_vel = {}; wall_indices = []

    bottom_wall_mask = pos[:, 1] < wall_width
    types[bottom_wall_mask] = 'W'; vel[bottom_wall_mask] = [-v_wall_b, 0.0]
    bottom_indices = np.where(bottom_wall_mask)[0]
    for idx in bottom_indices: wall_vel[idx] = np.array([-v_wall_b, 0.0])
    wall_indices.extend(bottom_indices)

    top_wall_mask = pos[:, 1] > L - wall_width; top_wall_mask &= (~bottom_wall_mask)
    types[top_wall_mask] = 'W'; vel[top_wall_mask] = [v_wall_b, 0.0]
    top_indices = np.where(top_wall_mask)[0]
    for idx in top_indices: wall_vel[idx] = np.array([v_wall_b, 0.0])
    wall_indices.extend(top_indices)

    molecule_indices = [idx for idx in molecule_indices if idx not in wall_indices]
    get_a_ij_func = get_a_ij_b; body_force = None
    print(f"  Created {num_chains_b} chains ({len(molecule_indices)} effective molecule particles).")
    print(f"  Added {n_fluid} fluid particles.")
    print(f"  Created {len(wall_indices)} wall particles.")
    return pos, vel, types, bonds, fixed_indices, wall_vel, get_a_ij_func, body_force, molecule_indices

def init_part_c(N, L, dt):
    """Initializes system for Part c: Poiseuille flow with rings."""
    print(f"Initializing Part c: N={N}, L={L}, dt={dt}")
    n_mol_particles = num_rings_c * ring_size_c
    if n_mol_particles > N: raise ValueError(f"Molecule particles ({n_mol_particles}) > N ({N})")

    types = []; bonds = []; molecule_indices = []
    start_idx = 0
    for _ in range(num_rings_c):
        ring_indices = list(range(start_idx, start_idx + ring_size_c))
        molecule_indices.extend(ring_indices)
        types.extend(['A'] * ring_size_c)
        for i in range(ring_size_c):
            j = (i + 1) % ring_size_c
            bonds.append(tuple(sorted((ring_indices[i], ring_indices[j]))))
        start_idx += ring_size_c

    n_fluid = N - n_mol_particles
    types.extend(['F'] * n_fluid); types = np.array(types)
    pos = np.random.rand(N, 2) * L; vel = np.zeros((N, 2))
    fixed_indices_list = []; wall_vel = None

    bottom_wall_mask = pos[:, 1] < wall_width
    types[bottom_wall_mask] = 'W'; fixed_indices_list.extend(np.where(bottom_wall_mask)[0])
    top_wall_mask = pos[:, 1] > L - wall_width; top_wall_mask &= (~bottom_wall_mask)
    types[top_wall_mask] = 'W'; fixed_indices_list.extend(np.where(top_wall_mask)[0])

    fixed_indices = np.array(fixed_indices_list, dtype=int); vel[fixed_indices] = 0.0
    molecule_indices = [idx for idx in molecule_indices if idx not in fixed_indices_list]
    get_a_ij_func = get_a_ij_c; body_force = F_body_c
    print(f"  Created {num_rings_c} rings ({len(molecule_indices)} effective molecule particles).")
    print(f"  Added {n_fluid} fluid particles.")
    print(f"  Created {len(fixed_indices)} fixed wall particles.")
    return pos, vel, types, bonds, fixed_indices, wall_vel, get_a_ij_func, body_force, molecule_indices

# ==============================================================================
# Analysis Functions
# ==============================================================================
def calculate_temperature(vel, m, fixed_indices=None, wall_vel=None):
    N = vel.shape[0]
    mobile_mask = np.ones(N, dtype=bool)
    if fixed_indices is not None and len(fixed_indices) > 0: mobile_mask[fixed_indices] = False
    if wall_vel is not None:
        wall_indices = list(wall_vel.keys())
        if len(wall_indices) > 0: mobile_mask[wall_indices] = False
    mobile_vel = vel[mobile_mask]
    if mobile_vel.shape[0] == 0: return 0.0
    kinetic_energy = 0.5 * m * np.sum(mobile_vel**2)
    N_mobile = mobile_vel.shape[0]; dof = N_mobile * 2
    if dof == 0: return 0.0
    return 2.0 * kinetic_energy / dof

def calculate_momentum(vel, m):
    return np.sum(vel * m, axis=0)

# ==============================================================================
# Visualization Functions
# ==============================================================================

# Keep static plots for Temperature, Velocity Profile, Final Distribution
def plot_temperature_history(temp_history, steps, dt, target_temp, title="Temperature Evolution", filename=None):
    plt.figure(figsize=(10, 5))
    time_axis = np.array(steps) * dt
    plt.plot(time_axis, temp_history, label="Measured Temperature")
    plt.axhline(target_temp, color='r', linestyle='--', label=f"Target T={target_temp:.4f}")
    plt.title(title); plt.xlabel(f"Time (units, dt={dt})"); plt.ylabel("Temperature (k_B T)")
    plt.legend(); plt.grid(True)
    if filename: plt.savefig(filename); print(f"Saved temperature plot to {filename}")
    plt.show()

def plot_velocity_profile(pos, vel, L, fixed_indices, wall_vel, num_bins=30, title="Velocity Profile", filename=None):
    N = pos.shape[0]; mobile_mask = np.ones(N, dtype=bool)
    if fixed_indices is not None and len(fixed_indices) > 0: mobile_mask[fixed_indices] = False
    if wall_vel is not None:
         wall_indices = list(wall_vel.keys())
         if len(wall_indices) > 0: mobile_mask[wall_indices] = False
    mobile_pos_y = pos[mobile_mask, 1]; mobile_vel_x = vel[mobile_mask, 0]
    if len(mobile_pos_y) == 0: print("Warning: No mobile particles for velocity profile."); return

    bin_edges = np.linspace(0, L, num_bins + 1); bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(mobile_pos_y, bin_edges[1:-1])
    avg_vel_x = np.zeros(num_bins); counts = np.zeros(num_bins, dtype=int)
    for i in range(len(mobile_vel_x)):
        bin_idx = bin_indices[i]
        if 0 <= bin_idx < num_bins: avg_vel_x[bin_idx] += mobile_vel_x[i]; counts[bin_idx] += 1
    valid_bins = counts > 0; avg_vel_x[valid_bins] /= counts[valid_bins]; avg_vel_x[~valid_bins] = np.nan

    plt.figure(figsize=(6, 8))
    plt.plot(avg_vel_x, bin_centers, marker='o', linestyle='-', label="Avg Vx Profile")
    plt.title(title); plt.xlabel("Average Velocity (Vx)"); plt.ylabel("Y Position")
    plt.ylim(0, L); plt.grid(True); plt.legend()
    if filename: plt.savefig(filename); print(f"Saved velocity profile to {filename}")
    plt.show()

def plot_molecule_distribution(pos, types, molecule_indices, L, axis=1, num_bins=30, title="Molecule Distribution", filename=None):
    if molecule_indices is None or len(molecule_indices) == 0: print("Warning: No molecule indices for distribution plot."); return
    mol_pos = pos[molecule_indices, axis]; axis_label = "Y" if axis == 1 else "X"
    plt.figure(figsize=(8, 5))
    plt.hist(mol_pos, bins=num_bins, range=(0, L), density=True, alpha=0.7, label=f"Molecule Dist ({axis_label})")
    plt.title(title); plt.xlabel(f"{axis_label} Position"); plt.ylabel("Density")
    plt.xlim(0, L); plt.grid(True, axis='y'); plt.legend()
    if filename: plt.savefig(filename); print(f"Saved molecule distribution plot to {filename}")
    plt.show()

# --- Animation Setup ---
def setup_animation_plot(L, title_prefix):
    """Sets up the figure and axes for animation."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle='--', alpha=0.5)
    # Use a placeholder scatter plot that will be updated
    scatter = ax.scatter([], [], s=5)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    title = ax.set_title(title_prefix) # Initial title
    return fig, ax, scatter, time_text, title

def update_animation_frame(frame_idx, pos_history, types, scatter, time_text, title, title_prefix, dt, frame_interval):
    """Updates the scatter plot for a single animation frame."""
    step = (frame_idx + 1) * frame_interval
    current_time = step * dt
    
    # Update particle positions
    scatter.set_offsets(pos_history[frame_idx])
    
    # Update colors (needed if types change, but they don't here after init)
    # particle_colors = [type_colors.get(ptype, default_color) for ptype in types]
    # scatter.set_facecolors(particle_colors) # Generally slow, avoid if colors are static

    # Update time text and title
    time_text.set_text(f'Time: {current_time:.2f}')
    title.set_text(f'{title_prefix} (Step {step})')
    
    # Return the updated artists
    return scatter, time_text, title


# ==============================================================================
# Simulation Loop
# ==============================================================================

def run_simulation(n_steps, dt, pos, vel, types, bonds, fixed_indices, wall_vel, get_a_ij_func, body_force, Ks=None, rs=None, L=L, m=m, rc=rc, rc_sq=rc_sq, gamma=gamma, sigma=sigma, kBT_target=kBT_target, print_freq=100, analysis_interval=100, record_video=False, video_filename="simulation.mp4", video_title_prefix="DPD Simulation"):
    """Runs the DPD simulation loop, stores history, and optionally records video."""
    N = pos.shape[0]
    print(f"\n--- Starting Simulation ---")
    print(f"N = {N}, Steps = {n_steps}, dt = {dt}")
    print(f"Target kBT = {kBT_target:.4f}")
    if fixed_indices is not None and len(fixed_indices)>0: print(f"Fixed particles = {len(fixed_indices)}")
    if wall_vel is not None: print(f"Moving wall particles = {len(wall_vel)}")
    if bonds: print(f"Bonds = {len(bonds)}, Ks = {Ks}, rs = {rs}")
    if body_force is not None: print(f"Body force = {body_force}")
    print("-" * 27)

    # History storage
    temp_history = []
    steps_history = []
    pos_history = [] # Store positions for video frames

    # --- Initial force calculation ---
    dpd_forces, _ = calculate_dpd_forces(pos, vel, types, L, rc, rc_sq, gamma, sigma, kBT_target, get_a_ij_func, dt)
    bond_forces = np.zeros_like(pos);
    if bonds and Ks is not None and rs is not None: bond_forces = calculate_bond_forces(pos, bonds, Ks, rs, L)
    total_body_force = np.zeros_like(pos)
    if body_force is not None:
        mobile_mask = np.ones(N, dtype=bool)
        if fixed_indices is not None and len(fixed_indices)>0: mobile_mask[fixed_indices] = False
        if wall_vel is not None: mobile_mask[list(wall_vel.keys())] = False
        total_body_force[mobile_mask] = body_force
    forces_t = dpd_forces + bond_forces + total_body_force

    # Record initial state for video if needed
    if record_video and 0 % VIDEO_FRAME_INTERVAL == 0:
         pos_history.append(np.copy(pos))

    # --- Simulation Loop ---
    start_time = time.time()
    for step in range(n_steps):
        pos, vel_intermediate = velocity_verlet_step(pos, vel, forces_t, dt, m, L, fixed_indices, wall_vel)
        dpd_forces_new, _ = calculate_dpd_forces(pos, vel_intermediate, types, L, rc, rc_sq, gamma, sigma, kBT_target, get_a_ij_func, dt)
        bond_forces_new = np.zeros_like(pos)
        if bonds and Ks is not None and rs is not None: bond_forces_new = calculate_bond_forces(pos, bonds, Ks, rs, L)
        forces_t_plus_dt = dpd_forces_new + bond_forces_new + total_body_force
        vel = velocity_update(vel, vel_intermediate, forces_t_plus_dt, dt, m, fixed_indices, wall_vel)
        forces_t = forces_t_plus_dt

        # --- Analysis & Output ---
        record_this_step = False
        if (step + 1) % analysis_interval == 0:
            temp = calculate_temperature(vel, m, fixed_indices, wall_vel)
            temp_history.append(temp)
            steps_history.append(step + 1)
            record_this_step = True # Also record analysis steps for convenience

        # --- Store data for video frame ---
        if record_video and (step + 1) % VIDEO_FRAME_INTERVAL == 0:
            if not record_this_step: # Avoid duplicate storage if analysis interval matches
                 pos_history.append(np.copy(pos))
            else: # Analysis step already calculated temp, just store pos
                 pos_history.append(np.copy(pos)) # Store position snapshot

        if (step + 1) % print_freq == 0:
            current_time = time.time(); elapsed = current_time - start_time
            temp_now = temp_history[-1] if temp_history else calculate_temperature(vel, m, fixed_indices, wall_vel)
            mom = calculate_momentum(vel, m)
            print(f"Step: {step+1}/{n_steps}, Temp: {temp_now:.4f}, Momentum: {mom}, Time: {elapsed:.2f}s")
            sys.stdout.flush()

    end_time = time.time()
    print(f"--- Simulation Finished ({end_time - start_time:.2f}s) ---")

    # --- Create and Save Animation ---
    if record_video and len(pos_history) > 0:
        print(f"Generating animation ({len(pos_history)} frames)...")
        fig, ax, scatter, time_text, title_obj = setup_animation_plot(L, video_title_prefix)
        # Set initial colors (assuming types don't change)
        particle_colors = [type_colors.get(ptype, default_color) for ptype in types]
        scatter.set_facecolors(particle_colors) # Set colors once

        # Create animation object
        # Need partial to pass extra arguments to update function
        update_func = partial(update_animation_frame, pos_history=pos_history, types=types,
                              scatter=scatter, time_text=time_text, title=title_obj,
                              title_prefix=video_title_prefix, dt=dt, frame_interval=VIDEO_FRAME_INTERVAL)

        anim = animation.FuncAnimation(fig, update_func, frames=len(pos_history), interval=50, blit=True)

        # Save the animation
        try:
            # Increase dpi for better quality if needed
            anim.save(video_filename, writer='ffmpeg', fps=VIDEO_FPS, dpi=150)
            print(f"Successfully saved animation to {video_filename}")
        except Exception as e:
            print(f"\n*** Error saving animation to {video_filename} ***")
            print("This usually means FFmpeg is not installed or not found.")
            print("Please install FFmpeg and ensure it's in your system's PATH.")
            print(f"Error details: {e}")
        plt.close(fig) # Close the figure after saving animation

    # Return final state and history
    results = {
        "pos": pos, "vel": vel, "types": types,
        "temp_history": temp_history, "steps_history": steps_history
    }
    return results

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    plt.close('all')

    # --- Part a: Fluid Test ---
    print("\n" + "="*30 + " Running Part a " + "="*30)
    n_steps_a_short = 2000
    n_steps_a_long = 10000 # Keep long run for dt=0.005
    results_a_long = None

    # Rerun short simulations first (no video for these quick tests)
    for dt_a in [0.02, 0.01]:
        pos_a, vel_a, types_a, bonds_a, fixed_a, wall_vel_a, get_a_ij_a, body_force_a, _ = init_part_a(N_a, L, dt_a)
        run_simulation(n_steps_a_short, dt_a, pos_a, vel_a, types_a, bonds_a, fixed_a, wall_vel_a, get_a_ij_a, body_force_a, print_freq=500, analysis_interval=100, record_video=False) # No video for short runs
        print(f"Part a finished for dt={dt_a}. Check momentum and temperature.")

    # Longer run for dt=0.005 WITH video
    dt_a = 0.005
    pos_a, vel_a, types_a, bonds_a, fixed_a, wall_vel_a, get_a_ij_a, body_force_a, _ = init_part_a(N_a, L, dt_a)
    results_a_long = run_simulation(
        n_steps_a_long, dt_a, pos_a, vel_a, types_a, bonds_a, fixed_a, wall_vel_a, get_a_ij_a, body_force_a,
        print_freq=1000, analysis_interval=100,
        record_video=True, video_filename="part_a_simulation.mp4", video_title_prefix="Part a (dt=0.005)"
    )
    print(f"Part a finished for dt={dt_a} (long run). Check momentum and temperature.")

    # Visualize Part a Temperature History (static plot)
    if results_a_long:
        plot_temperature_history(results_a_long["temp_history"], results_a_long["steps_history"], dt_a, kBT_target, title=f"Part a: Temperature Evolution (dt={dt_a})", filename="part_a_temp_history.png")


    # --- Part b: Couette Flow ---
    print("\n" + "="*30 + " Running Part b " + "="*30)
    n_steps_b = 5000
    dt_b = dt_default
    pos_b, vel_b, types_b, bonds_b, fixed_b, wall_vel_b, get_a_ij_b_func, body_force_b, mol_indices_b = init_part_b(N_b, L, dt_b)
    results_b = run_simulation(
        n_steps_b, dt_b, pos_b, vel_b, types_b, bonds_b, fixed_b, wall_vel_b, get_a_ij_b_func, body_force_b,
        Ks=Ks_b, rs=rs_b, print_freq=500, analysis_interval=100,
        record_video=True, video_filename="part_b_simulation.mp4", video_title_prefix="Part b (Couette)"
    )
    print("Part b finished.")

    # Visualize Part b (static plots for summary data)
    plot_velocity_profile(results_b["pos"], results_b["vel"], L, fixed_b, wall_vel_b, title="Part b: Couette Velocity Profile", filename="part_b_velocity_profile.png")
    plot_molecule_distribution(results_b["pos"], results_b["types"], mol_indices_b, L, axis=1, title="Part b: Chain Molecule Y-Distribution (Final)", filename="part_b_molecule_dist.png")


    # --- Part c: Poiseuille Flow ---
    print("\n" + "="*30 + " Running Part c " + "="*30)
    n_steps_c = 10000
    dt_c = dt_default
    pos_c, vel_c, types_c, bonds_c, fixed_c, wall_vel_c, get_a_ij_c_func, body_force_c_val, mol_indices_c = init_part_c(N_c, L, dt_c)
    results_c = run_simulation(
        n_steps_c, dt_c, pos_c, vel_c, types_c, bonds_c, fixed_c, wall_vel_c, get_a_ij_c_func, body_force_c_val,
        Ks=Ks_c, rs=rs_c, print_freq=1000, analysis_interval=100,
        record_video=True, video_filename="part_c_simulation.mp4", video_title_prefix="Part c (Poiseuille)"
    )
    print("Part c finished.")

    # Visualize Part c (static plots for summary data)
    plot_velocity_profile(results_c["pos"], results_c["vel"], L, fixed_c, wall_vel_c, title="Part c: Poiseuille Velocity Profile", filename="part_c_velocity_profile.png")
    plot_molecule_distribution(results_c["pos"], results_c["types"], mol_indices_c, L, axis=1, title="Part c: Ring Molecule Y-Distribution (Final)", filename="part_c_molecule_dist.png")

    print("="*30 + " End of Simulation " + "="*30)