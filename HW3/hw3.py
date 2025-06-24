import numpy as np
import itertools
import matplotlib.pyplot as plt # Optional: for plotting energy, RDF etc.
import multiprocessing as mp  # Add multiprocessing support
from functools import partial  # For partial function application in parallel processing

# --- Simulation Parameters (Reduced Units from Homework PDF) ---
L = 30.0  # Simulation box size
sigma = 1.0 # Lennard-Jones sigma
epsilon = 1.0 # Lennard-Jones epsilon
rc = 2.5 * sigma # Cut-off radius
rc2 = rc**2
m = 1.0 # Particle mass
k_B = 1.0 # Boltzmann constant (reduced units)

# --- Lennard-Jones Potential and Force ---
# Pre-calculate potential shift at cut-off
U_rc = 4.0 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)

def lj_potential_truncated(r2):
    """Calculates the truncated Lennard-Jones potential for squared distance r2."""
    if r2 > rc2:
        return 0.0
    sigma2_r2 = (sigma**2) / r2
    sigma6_r12 = sigma2_r2**3
    sigma12_r12 = sigma6_r12**2
    U = 4.0 * epsilon * (sigma12_r12 - sigma6_r12)
    return U - U_rc # Apply truncation shift

def lj_force(dr, r2):
    """Calculates the Lennard-Jones force vector for displacement vector dr and squared distance r2."""
    if r2 > rc2:
        return np.zeros(2)
    sigma2_r2 = (sigma**2) / r2
    sigma6_r12 = sigma2_r2**3
    sigma12_r12 = sigma6_r12**2
    # Force magnitude derivation F = -dU/dr, needs careful application of chain rule for F_x, F_y
    # F_mag = (48 * epsilon / r2) * (sigma12_r12 - 0.5 * sigma6_r12) # Magnitude based on formula
    # The formula in the PDF seems slightly different from the standard derivative, but let's use it:
    force_scalar = (48 * epsilon / r2) * (sigma12_r12 - 0.5 * sigma6_r12)
    # Force vector = scalar * (unit vector dr/r) = scalar * (dr / sqrt(r2))
    # force_vec = force_scalar * (dr / np.sqrt(r2)) # Standard way
    # However, the formula F_chi(r) = (48*chi/r^2) * [...] where chi is x or y seems to be missing a factor related to r in the denominator?
    # Let's assume F_chi(r) = F(r) * (chi/r) where F(r) is the magnitude.
    # F(r) = dU/dr = d/dr [ 4eps * ( (sig/r)^12 - (sig/r)^6 ) ]
    # F(r) = 4eps * [ -12 sig^12 / r^13 - (-6) sig^6 / r^7 ] * (-1) sign from F = -dU/dr
    # F(r) = 4eps * [ 12 sig^12 / r^13 - 6 sig^6 / r^7 ]
    # F(r) = 24eps/r * [ 2 (sig/r)^12 - (sig/r)^6 ]
    # Force vector = F(r) * (dr / r) = F(r)/r * dr
    # F(r)/r = 24eps/r^2 * [ 2 (sig/r)^12 - (sig/r)^6 ]
    # F(r)/r = (24 * epsilon / r2) * (2.0 * sigma12_r12 - sigma6_r12) # Force magnitude / r
    force_scalar_over_r = (24.0 * epsilon / r2) * (2.0 * sigma12_r12 - sigma6_r12)
    return force_scalar_over_r * dr # Force vector F = (F(r)/r) * dr

# --- Periodic Boundary Conditions ---
def apply_pbc(dr, box_size=L):
    """Applies minimum image convention."""
    return dr - box_size * np.rint(dr / box_size)

# --- Initialization ---
def initialize_particles(N, T_initial, box_size=L):
    """Initializes particle positions on a lattice and velocities."""
    # Positions on a square lattice
    n_side = int(np.ceil(np.sqrt(N)))
    spacing = box_size / n_side
    pos = np.zeros((N, 2))
    idx = 0
    for i in range(n_side):
        for j in range(n_side):
            if idx < N:
                pos[idx, 0] = (i + 0.5) * spacing
                pos[idx, 1] = (j + 0.5) * spacing
                idx += 1
    pos -= box_size / 2.0 # Center the lattice

    # Velocities (random, zero total momentum, scaled to T_initial)
    vel = np.random.rand(N, 2) - 0.5
    vel -= np.mean(vel, axis=0) # Ensure zero total momentum

    # Scale velocities to match initial temperature T_initial
    # Kinetic energy K = 0.5 * m * sum(v_i^2)
    # Temperature T = K / (N * d * 0.5 * k_B), where d=2 dimensions
    # T = (0.5 * m * sum(v_i^2)) / (N * 2 * 0.5 * k_B) = (m * sum(v_i^2)) / (2 * N * k_B)
    current_K = 0.5 * m * np.sum(vel**2)
    current_T = current_K / (N * k_B) # d=2, so N*d*0.5*kB = N*kB
    if current_T > 1e-6: # Avoid division by zero if velocities happen to be zero
        scale_factor = np.sqrt(T_initial / current_T)
        vel *= scale_factor
    else:
         # Handle the case where initial random velocities sum to nearly zero kinetic energy
         # Re-initialize or apply a small fixed magnitude velocity scaled appropriately
         print("Warning: Initial kinetic energy near zero. Applying small velocities.")
         vel = (np.random.rand(N, 2) - 0.5)
         vel -= np.mean(vel, axis=0)
         vel *= np.sqrt(2 * N * k_B * T_initial / (m * np.sum(vel**2)))


    # Verify zero momentum
    total_momentum = np.sum(m * vel, axis=0)
    # print(f"Initial total momentum: {total_momentum}") # Should be close to [0, 0]

    return pos, vel

# --- Cell List Implementation ---
class CellList:
    def __init__(self, box_size, cell_size):
        self.box_size = box_size
        self.cell_size = cell_size
        self.n_cells_side = int(np.floor(box_size / cell_size))
        if self.n_cells_side == 0:
             raise ValueError("Box size is smaller than cell size.")
        self.head = -np.ones((self.n_cells_side, self.n_cells_side), dtype=int)
        self.list = -np.ones(1, dtype=int) # Will be resized

    def build(self, pos):
        N = pos.shape[0]
        if self.list.shape[0] < N:
             self.list = -np.ones(N, dtype=int) # Resize if needed

        self.head.fill(-1) # Reset heads
        self.list[:N].fill(-1) # Reset list for current particles

        for i in range(N):
            # Map particle position to cell indices (assuming box centered at 0)
            cell_x = int(np.floor((pos[i, 0] + self.box_size / 2.0) / self.cell_size))
            cell_y = int(np.floor((pos[i, 1] + self.box_size / 2.0) / self.cell_size))

            # Handle particles exactly on the boundary or slightly outside due to float precision
            cell_x = max(0, min(cell_x, self.n_cells_side - 1))
            cell_y = max(0, min(cell_y, self.n_cells_side - 1))

            # Link particle into the cell list
            self.list[i] = self.head[cell_x, cell_y]
            self.head[cell_x, cell_y] = i

    def get_neighbors(self, particle_idx, pos):
        neighbors = []
        N = pos.shape[0]

        # Get cell of the current particle
        cell_x = int(np.floor((pos[particle_idx, 0] + self.box_size / 2.0) / self.cell_size))
        cell_y = int(np.floor((pos[particle_idx, 1] + self.box_size / 2.0) / self.cell_size))
        cell_x = max(0, min(cell_x, self.n_cells_side - 1))
        cell_y = max(0, min(cell_y, self.n_cells_side - 1))

        # Iterate through the particle's cell and neighboring cells (9 cells total)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell_x = (cell_x + dx) % self.n_cells_side
                neighbor_cell_y = (cell_y + dy) % self.n_cells_side

                current_neighbor_idx = self.head[neighbor_cell_x, neighbor_cell_y]
                while current_neighbor_idx != -1:
                    # Add if not the particle itself and within bounds
                    if current_neighbor_idx < N and current_neighbor_idx != particle_idx :
                         neighbors.append(current_neighbor_idx)
                    if current_neighbor_idx < N: # Bounds check before accessing self.list
                        current_neighbor_idx = self.list[current_neighbor_idx]
                    else:
                        # This case should ideally not happen if list is sized correctly
                        # print(f"Warning: Index {current_neighbor_idx} out of bounds for list size {N}")
                        break # Prevent potential infinite loop

        return neighbors

# --- Parallel Processing Helper Functions ---
def calculate_forces_chunk(particle_indices, pos, neighbors_dict, box_size=L):
    """Calculate forces for a subset of particles (used by parallel processing)."""
    N = pos.shape[0]
    chunk_forces = np.zeros((len(particle_indices), 2))
    chunk_potential = 0.0
    
    for chunk_idx, i in enumerate(particle_indices):
        for j_idx in neighbors_dict[i]:
            if j_idx > i:  # Avoid double counting and self-interaction
                dr = pos[i] - pos[j_idx]
                dr = apply_pbc(dr, box_size)
                r2 = np.sum(dr**2)
                
                if r2 < rc2:
                    chunk_potential += lj_potential_truncated(r2)
                    force_ij = lj_force(dr, r2)
                    chunk_forces[chunk_idx] += force_ij
    
    return chunk_forces, chunk_potential

# --- Optimized Parallel Energy Calculation ---
def calculate_energies_parallel(pos, vel, cell_list, box_size=L, n_cores=None):
    """Calculates energies and forces using parallel processing."""
    N = pos.shape[0]
    
    # Determine number of cores to use
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    # Only use parallel for larger particle counts
    if N < 200 or n_cores <= 1:
        return calculate_energies(pos, vel, cell_list, box_size)
    
    # Build cell list and get all neighbors
    cell_list.build(pos)
    
    # Pre-compute all neighbors (this avoids duplicating work in each process)
    neighbors_dict = {}
    for i in range(N):
        neighbors_dict[i] = cell_list.get_neighbors(i, pos)
    
    # Split particles into chunks for parallel processing
    chunk_size = max(1, N // n_cores)
    particle_chunks = [list(range(i, min(i+chunk_size, N))) for i in range(0, N, chunk_size)]
    
    # Create a partial function with fixed arguments
    partial_forces_func = partial(calculate_forces_chunk, 
                                  pos=pos, 
                                  neighbors_dict=neighbors_dict, 
                                  box_size=box_size)
    
    # Calculate forces in parallel
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(partial_forces_func, particle_chunks)
    
    # Combine results from all processes
    forces = np.zeros((N, 2))
    potential_energy = 0.0
    
    for chunk_idx, (chunk_forces, chunk_potential) in enumerate(results):
        chunk_particles = particle_chunks[chunk_idx]
        forces[chunk_particles] = chunk_forces
        potential_energy += chunk_potential
    
    # Apply Newton's third law (forces[j] -= force_ij)
    for i in range(N):
        for j_idx in neighbors_dict[i]:
            if j_idx > i:  # Only process one direction
                dr = pos[i] - pos[j_idx]
                dr = apply_pbc(dr, box_size)
                r2 = np.sum(dr**2)
                
                if r2 < rc2:
                    force_ij = lj_force(dr, r2)
                    forces[j_idx] -= force_ij
    
    # Calculate kinetic energy
    kinetic_energy = 0.5 * m * np.sum(vel**2)
    total_energy = kinetic_energy + potential_energy
    
    return potential_energy, kinetic_energy, total_energy, forces

# --- Energy Calculation ---
def calculate_energies(pos, vel, cell_list, box_size=L):
    N = pos.shape[0]
    potential_energy = 0.0
    forces = np.zeros((N, 2))

    cell_list.build(pos) # Update cell list for current positions

    for i in range(N):
        neighbors = cell_list.get_neighbors(i, pos)
        for j_idx in neighbors:
             if j_idx > i: # Avoid double counting and self-interaction
                dr = pos[i] - pos[j_idx]
                dr = apply_pbc(dr, box_size)
                r2 = np.sum(dr**2)

                if r2 < rc2:
                    potential_energy += lj_potential_truncated(r2)
                    force_ij = lj_force(dr, r2)
                    forces[i] += force_ij
                    forces[j_idx] -= force_ij # Newton's third law

    kinetic_energy = 0.5 * m * np.sum(vel**2)
    total_energy = kinetic_energy + potential_energy
    return potential_energy, kinetic_energy, total_energy, forces

# --- Radial Distribution Function (RDF) ---
def calculate_rdf(pos, N, box_size, dr_bin, max_r):
    n_bins = int(max_r / dr_bin)
    rdf_hist = np.zeros(n_bins)
    rdf_bins = np.linspace(0, max_r, n_bins + 1)
    pair_count = 0

    # Determine cell size needed for RDF calculation up to max_r
    # If using cell list for RDF, ensure cell_size > max_r or modify neighbour search
    # For simplicity here, calculate RDF with all pairs (slower for large N)
    # If max_r > rc, cell list needs adjustment or use brute force below
    rdf_cell_size = max(rc, max_r) # Ensure cell list covers RDF range if used
    # rdf_cell_list = CellList(box_size, rdf_cell_size) # Optional: use cell list
    # rdf_cell_list.build(pos)

    for i in range(N):
        # neighbours = rdf_cell_list.get_neighbours(i, pos) # If using cell list
        # for j_idx in neighbours: # If using cell list
        for j_idx in range(N): # Brute force - check all pairs
            if j_idx > i: # Avoid double counting
                dr = pos[i] - pos[j_idx]
                dr = apply_pbc(dr, box_size)
                r = np.sqrt(np.sum(dr**2))

                if r < max_r:
                    bin_index = int(r / dr_bin)
                    if bin_index < n_bins:
                        rdf_hist[bin_index] += 2 # Count pair twice (i->j and j->i)
                        pair_count +=2


    # Normalize RDF
    volume = box_size**2
    particle_density = N / volume
    # Normalization factor for each bin
    bin_volumes = np.pi * (rdf_bins[1:]**2 - rdf_bins[:-1]**2)
    ideal_gas_counts = particle_density * bin_volumes

    # Avoid division by zero for bins with zero volume or zero ideal count
    rdf = np.zeros(n_bins)
    valid_bins = ideal_gas_counts > 1e-9
    # The normalization factor should be N_pairs * ideal_gas_counts[bin] / Volume?
    # Standard normalization: g(r) = V * N(r) / (N_particles * N_neighbours_ideal(r))
    # N(r) = count in bin dr at distance r
    # N_neighbours_ideal(r) = density * shell_volume = (N/V) * 2*pi*r*dr (in 2D)
    # g(r) = V * rdf_hist[bin] / (N * (N/V) * 2*pi*r*dr) = rdf_hist[bin] / (N * density * 2*pi*r*dr)
    # Let's use the formula: g(r) = histogram(r) / (N * density * V_shell(r)) where N is total pairs considered? No, N particles.
    # Normalization: rdf = hist / (N_particles * density * shell_volume)
    # Shell volume in 2D: pi*((r+dr)^2 - r^2) = pi*(r^2 + 2r*dr + dr^2 - r^2) approx 2*pi*r*dr for small dr
    bin_centers = (rdf_bins[:-1] + rdf_bins[1:]) / 2.0
    # Correct normalization factor: (Volume / (N * (N-1))) * (Counts_in_bin / Shell_Volume_of_bin) - for distinct pairs i!=j
    # Let's use the common definition: number density in shell / bulk density
    # Number density in shell = counts / shell_volume
    # g(r) = (counts / shell_volume) / bulk_density = (rdf_hist / bin_volumes) / particle_density
    rdf[valid_bins] = (rdf_hist[valid_bins] / bin_volumes[valid_bins]) / particle_density

    # Alternative normalization often seen: scale by N_pairs/(Area*rho*2*pi*r*dr)
    # Let's stick to the density ratio definition. Need total number of pairs counted.
    # Normalization check: rdf should go to 1 at large r for fluids.

    # Correct normalization: g(r) = V/(N*(N-1)) * counts(r) / (2*pi*r*dr)  -- using N*(N-1)/2 pairs
    # Let's try: g(r) = <n(r)> / (rho * 2*pi*r*dr) where <n(r)> is avg number in shell
    # Avg number = total counts / N
    avg_counts_in_shell = rdf_hist / N if N > 0 else np.zeros(n_bins)
    shell_volumes_approx = 2 * np.pi * bin_centers * dr_bin # Approximate shell volume
    valid = (particle_density > 1e-9) & (shell_volumes_approx > 1e-9)
    rdf[valid] = avg_counts_in_shell[valid] / (particle_density * shell_volumes_approx[valid])


    return rdf, bin_centers

# --- Berendsen Thermostat ---
def apply_berendsen_thermostat(vel, T_target, dt, tau, current_K, N):
    """Applies the Berendsen thermostat scaling."""
    # Calculate current temperature
    # T = K / (N*dof_per_particle*0.5*kB) = K / (N*dims*0.5*kB)
    # dof_per_particle = dims = 2
    current_T = current_K / (N * k_B) # Assuming d=2, k_B=1

    if current_T > 1e-6 and tau > 1e-9: # Avoid division by zero or invalid tau
         # Scaling factor lambda^2 = 1 + (dt/tau) * (T_target/T_current - 1)
         lambda_sq = 1.0 + (dt / tau) * (T_target / current_T - 1.0)
         # Prevent overly large scaling (instability)
         if lambda_sq < 0:
              print(f"Warning: Negative lambda_sq ({lambda_sq}) in Berendsen thermostat. Clamping to 0.")
              lambda_sq = 0.0
         scale_factor = np.sqrt(lambda_sq)
         vel *= scale_factor
    # Else: No scaling if T=0 or tau=0
    return vel

# --- Velocity-Verlet Integration ---
def velocity_verlet_step(pos, vel, force, dt, m=1.0):
    # Update velocities part 1: v(t + dt/2) = v(t) + 0.5 * a(t) * dt
    accel = force / m
    vel += 0.5 * accel * dt

    # Update positions: r(t + dt) = r(t) + v(t + dt/2) * dt
    pos += vel * dt

    # Apply periodic boundary conditions to positions
    pos = pos - L * np.rint(pos / L) # More robust PBC wrap

    # Forces and energies at r(t + dt) will be calculated next
    # Update velocities part 2: v(t + dt) = v(t + dt/2) + 0.5 * a(t + dt) * dt
    # This part requires the force at the new position, calculated *after* this function returns
    return pos, vel

# --- Main Simulation Loop ---
def run_simulation(N, T_initial, dt, n_steps, use_thermostat=False, T_target=None, tau=None, 
                  rdf_interval=100, rdf_max_r=5.0, rdf_dr=0.05, use_parallel=True, n_cores=None):

    pos, vel = initialize_particles(N, T_initial)
    cell_list = CellList(L, rc) # Initialize cell list with cutoff radius

    # Determine if we should use parallel processing
    parallel_enabled = use_parallel and N >= 200
    if parallel_enabled:
        if n_cores is None:
            n_cores = max(1, mp.cpu_count() - 1)
        print(f"Using parallel processing with {n_cores} cores for N={N} particles")
    
    # --- Data Storage ---
    times = np.arange(n_steps + 1) * dt
    total_energies = np.zeros(n_steps + 1)
    kinetic_energies = np.zeros(n_steps + 1)
    potential_energies = np.zeros(n_steps + 1)
    temperatures = np.zeros(n_steps + 1)
    total_momenta = np.zeros((n_steps + 1, 2))
    rdf_results = {} # Store RDF at specific steps

    # --- Initial State (t=0) ---
    if parallel_enabled:
        U, K, E_tot, F = calculate_energies_parallel(pos, vel, cell_list, L, n_cores)
    else:
        U, K, E_tot, F = calculate_energies(pos, vel, cell_list)
    potential_energies[0] = U
    kinetic_energies[0] = K
    total_energies[0] = E_tot
    temperatures[0] = K / (N * k_B) # T = K / (N*dof*0.5*kB) = K/(N*1*1) for d=2, kB=1
    total_momenta[0] = np.sum(m * vel, axis=0)

    print(f"Starting simulation: N={N}, T_initial~={T_initial:.2f}, dt={dt}, Steps={n_steps}")
    print(f"Thermostat: {'ON' if use_thermostat else 'OFF'}, T_target={T_target}, tau={tau}")
    print(f"Initial State: K={K:.4f}, U={U:.4f}, E_tot={E_tot:.4f}, T={temperatures[0]:.4f}, P={total_momenta[0]}")

    # --- Simulation Steps ---
    for step in range(n_steps):
        # Velocity Verlet Step 1 & 2 (Positions and half-step velocity)
        pos, vel = velocity_verlet_step(pos, vel, F, dt, m)
        
        # Calculate Forces and Energies at new position r(t+dt)
        if parallel_enabled:
            U, K, E_tot, F_new = calculate_energies_parallel(pos, vel, cell_list, L, n_cores)
        else:
            U, K, E_tot, F_new = calculate_energies(pos, vel, cell_list)
        
        # Velocity Verlet Step 3 (Final velocity update)
        accel_new = F_new / m
        vel += 0.5 * accel_new * dt

        # Apply Thermostat (if enabled)
        if use_thermostat and T_target is not None and tau is not None:
             # Recalculate K after final velocity update if thermostat applied after?
             # Usually thermostat is applied *before* the final velocity update, or using K from v(t+dt/2)?
             # Let's apply it *after* the full step, using the K derived from v(t+dt)
             current_K_for_thermo = 0.5 * m * np.sum(vel**2)
             vel = apply_berendsen_thermostat(vel, T_target, dt, tau, current_K_for_thermo, N)
             # Update kinetic energy after thermostat scaling for recording
             K = 0.5 * m * np.sum(vel**2)
             E_tot = K + U # Recalculate total energy after thermostat

        # --- Store Data ---
        potential_energies[step + 1] = U
        kinetic_energies[step + 1] = K
        total_energies[step + 1] = E_tot
        temperatures[step + 1] = K / (N * k_B)
        total_momenta[step + 1] = np.sum(m * vel, axis=0)

        # --- Calculate RDF Periodically (after potential equilibration) ---
        # Example: start calculating RDF after 1000 steps and every 'rdf_interval' steps
        equilibration_steps = 1000 # Estimate, adjust based on energy plots
        if step > equilibration_steps and (step + 1) % rdf_interval == 0:
            rdf, bins = calculate_rdf(pos, N, L, rdf_dr, rdf_max_r)
            rdf_results[step + 1] = (bins, rdf)
            # print(f"Step {step+1}: Calculated RDF.")


        # --- Progress Output ---
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{n_steps}: K={K:.4f}, U={U:.4f}, E_tot={E_tot:.4f}, T={temperatures[step+1]:.4f}, P={total_momenta[step+1]}")

    print("Simulation finished.")

    # --- Return Results ---
    results = {
        "times": times,
        "total_energy": total_energies,
        "kinetic_energy": kinetic_energies,
        "potential_energy": potential_energies,
        "temperature": temperatures,
        "total_momentum": total_momenta,
        "final_pos": pos,
        "final_vel": vel,
        "rdf_results": rdf_results,
        "N": N,
        "L": L,
        "dt": dt,
        "T_initial": T_initial,
        "use_thermostat": use_thermostat,
        "T_target": T_target
    }
    return results

# --- Function to Plot Results (Example) ---
def plot_results(results):
    print("\n--- Plotting Results ---")
    plt.figure(figsize=(12, 10))

    # Energy Plots
    plt.subplot(2, 2, 1)
    plt.plot(results["times"], results["total_energy"], label='Total Energy')
    plt.plot(results["times"], results["kinetic_energy"], label='Kinetic Energy')
    plt.plot(results["times"], results["potential_energy"], label='Potential Energy')
    plt.xlabel("Time (reduced units)")
    plt.ylabel("Energy (reduced units)")
    plt.title("Energy Evolution")
    plt.legend()
    plt.grid(True)

    # Temperature Plot
    plt.subplot(2, 2, 2)
    plt.plot(results["times"], results["temperature"], label='Temperature')
    if results["use_thermostat"] and results["T_target"] is not None:
        plt.axhline(results["T_target"], color='r', linestyle='--', label=f'Target T ({results["T_target"]})')
    plt.xlabel("Time (reduced units)")
    plt.ylabel("Temperature (reduced units)")
    plt.title("Temperature Evolution")
    plt.legend()
    plt.grid(True)

    # Momentum Conservation Check
    plt.subplot(2, 2, 3)
    plt.plot(results["times"], results["total_momentum"][:, 0], label='Total Momentum X')
    plt.plot(results["times"], results["total_momentum"][:, 1], label='Total Momentum Y')
    plt.xlabel("Time (reduced units)")
    plt.ylabel("Momentum (reduced units)")
    plt.title("Total Momentum Conservation")
    plt.legend()
    plt.grid(True)
    # Check Y-axis scale - should be very close to zero
    max_mom = np.max(np.abs(results["total_momentum"]))
    plt.ylim(-max_mom*2 if max_mom > 1e-9 else -1e-9, max_mom*2 if max_mom > 1e-9 else 1e-9)


    # RDF Plot (last calculated RDF)
    plt.subplot(2, 2, 4)
    if results["rdf_results"]:
        last_step = max(results["rdf_results"].keys())
        bins, rdf = results["rdf_results"][last_step]
        plt.plot(bins, rdf, label=f'RDF at step {last_step}')
        plt.xlabel("r (reduced units)")
        plt.ylabel("g(r)")
        plt.title("Radial Distribution Function")
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "No RDF data calculated/stored", ha='center', va='center')


    plt.tight_layout()
    plt.show()

# --- Simulation Setup & Execution ---

# --- Analysis Functions ---
def calculate_energy_fluctuations(energy_array):
    """Calculate the standard deviation of energy to quantify fluctuations."""
    # Skip initial equilibration period (first 20%)
    eq_start = int(len(energy_array) * 0.2)
    eq_energy = energy_array[eq_start:]
    mean_energy = np.mean(eq_energy)
    std_energy = np.std(eq_energy)
    rel_fluctuation = std_energy / abs(mean_energy) if abs(mean_energy) > 1e-10 else 0
    return mean_energy, std_energy, rel_fluctuation

def estimate_equilibration_time(energy_array, times_array, window_size=100):
    """Estimate equilibration time by analyzing when energy fluctuations stabilize."""
    if len(energy_array) < window_size * 2:
        return None  # Not enough data
    
    # Calculate rolling standard deviation
    rolling_std = np.zeros(len(energy_array) - window_size)
    for i in range(len(rolling_std)):
        rolling_std[i] = np.std(energy_array[i:i+window_size])
    
    # Find where the rolling std stabilizes (within 10% of final value)
    final_std = np.mean(rolling_std[-int(len(rolling_std)/5):])  # Average of last 20%
    for i in range(len(rolling_std)):
        if abs(rolling_std[i] - final_std) / final_std < 0.1:
            return times_array[i]
    
    return times_array[-1]  # Default to end of simulation if no equilibration detected

def save_simulation_data(results, filename_prefix):
    """Save simulation data to files for later analysis."""
    np.savez(f"{filename_prefix}_data.npz", 
             times=results["times"],
             total_energy=results["total_energy"],
             kinetic_energy=results["kinetic_energy"],
             potential_energy=results["potential_energy"],
             temperature=results["temperature"],
             total_momentum=results["total_momentum"],
             N=results["N"],
             L=results["L"],
             dt=results["dt"],
             T_initial=results["T_initial"])
    
    # Save final RDF if available
    if results["rdf_results"]:
        last_step = max(results["rdf_results"].keys())
        bins, rdf = results["rdf_results"][last_step]
        np.savez(f"{filename_prefix}_rdf.npz", bins=bins, rdf=rdf, step=last_step)
    
    print(f"Data saved to {filename_prefix}_data.npz")

def plot_enhanced_results(results, title="Simulation Results", show_equilibration=True, save_fig=False, filename=None):
    """Enhanced plotting function with more analysis."""
    print(f"\n--- Plotting {title} ---")
    fig = plt.figure(figsize=(15, 12))
    
    # Energy Plots with fluctuation analysis
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(results["times"], results["total_energy"], label='Total Energy')
    ax1.plot(results["times"], results["kinetic_energy"], label='Kinetic Energy')
    ax1.plot(results["times"], results["potential_energy"], label='Potential Energy')
    
    if show_equilibration:
        # Estimate equilibration time
        eq_time_K = estimate_equilibration_time(results["kinetic_energy"], results["times"])
        eq_time_U = estimate_equilibration_time(results["potential_energy"], results["times"])
        eq_time_E = estimate_equilibration_time(results["total_energy"], results["times"])
        
        if eq_time_K is not None:
            ax1.axvline(eq_time_K, color='r', linestyle='--', label=f'K Equilibration (~{eq_time_K:.1f})')
        if eq_time_U is not None:
            ax1.axvline(eq_time_U, color='g', linestyle='--', label=f'U Equilibration (~{eq_time_U:.1f})')
        if eq_time_E is not None:
            ax1.axvline(eq_time_E, color='b', linestyle=':', label=f'E Equilibration (~{eq_time_E:.1f})')
    
    # Calculate energy fluctuations
    mean_E, std_E, rel_fluc_E = calculate_energy_fluctuations(results["total_energy"])
    ax1.set_title(f"Energy Evolution (Fluctuation: {rel_fluc_E:.2e})")
    ax1.set_xlabel("Time (reduced units)")
    ax1.set_ylabel("Energy (reduced units)")
    ax1.legend()
    ax1.grid(True)
    
    # Temperature Plot
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(results["times"], results["temperature"], label='Temperature')
    if results["use_thermostat"] and results["T_target"] is not None:
        ax2.axhline(results["T_target"], color='r', linestyle='--', label=f'Target T ({results["T_target"]})')
        
        # Estimate temperature equilibration for NVT
        eq_time_T = estimate_equilibration_time(results["temperature"], results["times"])
        if eq_time_T is not None and show_equilibration:
            ax2.axvline(eq_time_T, color='g', linestyle='--', label=f'T Equilibration (~{eq_time_T:.1f})')
    
    ax2.set_xlabel("Time (reduced units)")
    ax2.set_ylabel("Temperature (reduced units)")
    ax2.set_title("Temperature Evolution")
    ax2.legend()
    ax2.grid(True)
    
    # Momentum Conservation Check
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(results["times"], results["total_momentum"][:, 0], label='Total Momentum X')
    ax3.plot(results["times"], results["total_momentum"][:, 1], label='Total Momentum Y')
    ax3.set_xlabel("Time (reduced units)")
    ax3.set_ylabel("Momentum (reduced units)")
    ax3.set_title("Total Momentum Conservation")
    ax3.legend()
    ax3.grid(True)
    
    # Check Y-axis scale - should be very close to zero
    max_mom = np.max(np.abs(results["total_momentum"]))
    ax3.set_ylim(-max_mom*2 if max_mom > 1e-9 else -1e-9, max_mom*2 if max_mom > 1e-9 else 1e-9)
    
    # Energy fluctuation as function of time
    ax4 = plt.subplot(3, 2, 4)
    window_size = min(100, max(10, len(results["times"])//20))  # More conservative window size
    if window_size > 0 and len(results["times"]) > window_size*2:
        # Calculate rolling standard deviation more efficiently
        rolling_energy_std = np.zeros(len(results["times"]) - window_size)
        total_energy = results["total_energy"]
        
        for i in range(len(rolling_energy_std)):
            rolling_energy_std[i] = np.std(total_energy[i:i+window_size])
            
        # Ensure arrays have matching lengths
        rolling_times = results["times"][window_size//2:-(window_size-window_size//2)]
        
        # Double-check lengths before plotting
        if len(rolling_energy_std) > 0:
            if len(rolling_energy_std) != len(rolling_times):
                # Trim to match the shorter length if needed
                min_len = min(len(rolling_energy_std), len(rolling_times))
                rolling_energy_std = rolling_energy_std[:min_len]
                rolling_times = rolling_times[:min_len]
                
            ax4.plot(rolling_times, rolling_energy_std)
            ax4.set_xlabel("Time (reduced units)")
            ax4.set_ylabel("Energy Std Dev (window)")
            ax4.set_title(f"Energy Fluctuations Over Time (window={window_size})")
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, "Insufficient data for energy fluctuation analysis", 
                    ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, "Insufficient data for energy fluctuation analysis", 
                ha='center', va='center')
    
    # RDF Plot (last calculated RDF)
    ax5 = plt.subplot(3, 2, 5)
    if results["rdf_results"]:
        last_step = max(results["rdf_results"].keys())
        bins, rdf = results["rdf_results"][last_step]
        ax5.plot(bins, rdf, label=f'RDF at step {last_step}')
        ax5.set_xlabel("r (reduced units)")
        ax5.set_ylabel("g(r)")
        ax5.set_title("Radial Distribution Function")
        ax5.legend()
        ax5.grid(True)
    else:
        ax5.text(0.5, 0.5, "No RDF data calculated/stored", ha='center', va='center')
    
    # Final particle positions plot
    ax6 = plt.subplot(3, 2, 6)
    if "final_pos" in results:
        ax6.scatter(results["final_pos"][:, 0], results["final_pos"][:, 1], s=10, alpha=0.7)
        ax6.set_xlim(-L/2, L/2)
        ax6.set_ylim(-L/2, L/2)
        ax6.set_xlabel("x (reduced units)")
        ax6.set_ylabel("y (reduced units)")
        ax6.set_title(f"Final Particle Positions (N={results['N']})")
        ax6.grid(True)
        
        # Add box boundaries
        box_x = [-L/2, L/2, L/2, -L/2, -L/2]
        box_y = [-L/2, -L/2, L/2, L/2, -L/2]
        ax6.plot(box_x, box_y, 'k--', alpha=0.5)
    
    plt.suptitle(f"{title} (N={results['N']}, dt={results['dt']}, T_init={results['T_initial']})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_fig and filename:
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}.png")
    
    plt.show()

# --- Functions to Run Simulation Sets ---

def run_nve_simulations():
    """Run NVE simulations with different parameters for part (a)."""
    print("\n=== Running NVE Simulations (Part a) ===")
    
    # Parameters to test
    n_values = [100, 400, 900]  # Different particle counts
    dt_values = [0.001, 0.005, 0.01]  # Different time steps
    velocity_seeds = [42, 123, 789]  # Seeds for different initial velocities
    
    # Store results
    all_results = {}
    
    # Run simulations with different particle counts
    for N in n_values:
        for dt in dt_values:
            # Use default initial velocity (original seed)
            print(f"\nRunning NVE simulation with N={N}, dt={dt}")
            np.random.seed(42)  # Reset seed for reproducibility
            steps = int(100 / dt)  # Ensure similar simulation duration
            
            # Enable parallel processing for large systems
            use_parallel = N >= 200
            n_cores = max(1, mp.cpu_count() - 1) if use_parallel else None
            
            results = run_simulation(
                N=N, 
                T_initial=0.5,  # Moderate initial temperature
                dt=dt, 
                n_steps=steps,
                use_thermostat=False,
                rdf_interval=steps // 10,  # Calculate RDF several times during simulation
                use_parallel=use_parallel,
                n_cores=n_cores
            )
            
            # Calculate energy fluctuations
            mean_E, std_E, rel_fluc_E = calculate_energy_fluctuations(results["total_energy"])
            print(f"Energy fluctuations: mean={mean_E:.4f}, std={std_E:.4f}, rel={rel_fluc_E:.4e}")
            
            # Estimate equilibration time
            eq_time_K = estimate_equilibration_time(results["kinetic_energy"], results["times"])
            eq_time_U = estimate_equilibration_time(results["potential_energy"], results["times"])
            print(f"Estimated equilibration times: K={eq_time_K:.2f}, U={eq_time_U:.2f}")
            
            # Save and plot results
            key = f"N{N}_dt{dt}"
            all_results[key] = results
            save_simulation_data(results, f"nve_{key}")
            plot_enhanced_results(
                results, 
                title=f"NVE Simulation (N={N}, dt={dt})", 
                save_fig=True, 
                filename=f"nve_{key}"
            )
    
    # Run simulations with different initial velocities for one set of parameters
    N = 400  # Middle value of N
    dt = 0.005  # Middle value of dt
    for seed in velocity_seeds:
        print(f"\nRunning NVE simulation with N={N}, dt={dt}, seed={seed}")
        np.random.seed(seed)  # Set seed for different initial velocities
        steps = int(100 / dt)
        
        # Enable parallel processing
        use_parallel = True  # N=400 is large enough
        n_cores = max(1, mp.cpu_count() - 1)
        
        results = run_simulation(
            N=N, 
            T_initial=0.5,
            dt=dt, 
            n_steps=steps,
            use_thermostat=False,
            rdf_interval=steps // 10,
            use_parallel=use_parallel,
            n_cores=n_cores
        )
        
        # Save and plot results
        key = f"N{N}_dt{dt}_seed{seed}"
        all_results[key] = results
        save_simulation_data(results, f"nve_{key}")
        plot_enhanced_results(
            results, 
            title=f"NVE Simulation (N={N}, dt={dt}, seed={seed})",
            save_fig=True, 
            filename=f"nve_{key}"
        )
    
    return all_results

def run_nvt_simulations():
    """Run NVT simulations with specified parameters for part (b)."""
    print("\n=== Running NVT Simulations (Part b) ===")
    
    # Required parameter sets
    param_sets = [
        {"N": 100, "T": 0.1},
        {"N": 100, "T": 1.0},
        {"N": 625, "T": 1.0},
        {"N": 900, "T": 1.0}
    ]
    
    dt = 0.005  # Use stable dt from part (a)
    steps = 10000  # Longer simulations for better equilibration
    
    # Store results
    all_results = {}
    
    for params in param_sets:
        N = params["N"]
        T = params["T"]
        
        print(f"\nRunning NVT simulation with N={N}, T={T}")
        np.random.seed(42)  # Reset seed for reproducibility
        
        # Berendsen thermostat parameters
        tau = dt / 0.0025  # As per problem statement dt/tau = 0.0025
        
        # Enable parallel processing for large systems and determine core count
        use_parallel = N >= 200
        n_cores = max(1, mp.cpu_count() - 1) if use_parallel else None
        
        results = run_simulation(
            N=N, 
            T_initial=T,  # Start near target temperature
            dt=dt, 
            n_steps=steps,
            use_thermostat=True, 
            T_target=T, 
            tau=tau,
            rdf_interval=100,  # Calculate RDF frequently after equilibration
            use_parallel=use_parallel,
            n_cores=n_cores
        )
        
        # Estimate temperature equilibration time
        eq_time_T = estimate_equilibration_time(results["temperature"], results["times"])
        print(f"Estimated temperature equilibration time: {eq_time_T:.2f}")
        
        # Save and plot results
        key = f"N{N}_T{T}"
        all_results[key] = results
        save_simulation_data(results, f"nvt_{key}")
        plot_enhanced_results(
            results, 
            title=f"NVT Simulation (N={N}, T={T})", 
            save_fig=True, 
            filename=f"nvt_{key}"
        )
    
    return all_results

# Run all simulations needed for the report
if __name__ == "__main__":
    print("Running simulations for homework analysis...")
    print(f"Number of CPU cores available: {mp.cpu_count()}")
    
    # Part (a): NVE simulations with different parameters
    nve_results = run_nve_simulations()
    
    # Part (b): NVT simulations with specified parameters
    nvt_results = run_nvt_simulations()
    
    print("\nAll simulations completed. Results saved for report analysis.")

# --- END OF PROGRAM ---