import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
J = 1  
k_B = 1  
temperatures = [0.01, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0]
system_sizes = [5, 10, 15]
N_therm = 100000  # Thermalization steps per homework
N_sample = 5000  
possible_dE = [-8, -4, 0, 4, 8]  # Precompute acceptance probabilities for these dE

# --- Functions ---
def initialize_lattice(L):
    return np.random.choice([-1, 1], size=(L, L))

def compute_energy(spins, L):
    energy = 0.0
    for i in range(L):
        for j in range(L):
            S = spins[i, j]
            right = spins[(i+1) % L, j]
            down = spins[i, (j+1) % L]
            energy += (-J * S * right) + (-J * S * down)
    return energy / 2  # Correct for double-counting

def compute_magnetization(spins):
    return np.sum(spins) / spins.size

def precompute_accept_prob(T):
    beta = 1 / (k_B * T) if T != 0 else float('inf')
    accept_prob = {}
    for de in possible_dE:
        if de <= 0:
            accept_prob[de] = 1.0
        else:
            accept_prob[de] = np.exp(-beta * de)
    return accept_prob

def simulate(T, L):
    accept_prob = precompute_accept_prob(T)
    spins = initialize_lattice(L)
    # Thermalization
    for _ in range(N_therm):
        for _ in range(L*L):
            i, j = np.random.randint(0, L, 2)
            S = spins[i, j]
            neighbors = spins[(i+1)%L, j] + spins[i, (j+1)%L] + spins[(i-1)%L, j] + spins[i, (j-1)%L]
            dE = 2 * J * S * neighbors
            prob = accept_prob.get(dE, np.exp(-dE/(k_B*T)) if dE > 0 else 1.0)
            if np.random.rand() < prob:
                spins[i, j] *= -1
    # Sampling
    energies = []
    magnetizations = []
    for _ in range(N_sample):
        for _ in range(L*L):
            i, j = np.random.randint(0, L, 2)
            S = spins[i, j]
            neighbors = spins[(i+1)%L, j] + spins[i, (j+1)%L] + spins[(i-1)%L, j] + spins[i, (j-1)%L]
            dE = 2 * J * S * neighbors
            prob = accept_prob.get(dE, np.exp(-dE/(k_B*T)) if dE > 0 else 1.0)
            if np.random.rand() < prob:
                spins[i, j] *= -1
        energies.append(compute_energy(spins, L))
        magnetizations.append(abs(compute_magnetization(spins)))
    return np.mean(energies), np.mean(magnetizations)

# Part a & b: System size dependence
for L in system_sizes:
    avg_energies = []
    avg_magnetizations = []
    for T in temperatures:
        E, M = simulate(T, L)
        avg_energies.append(E)
        avg_magnetizations.append(M)
        print(f"T={T}: ⟨E⟩={E:.2f}, ⟨|M|⟩={M:.2f}")

    # Plotting
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(temperatures, avg_energies, 'o-')
    plt.xlabel('Temperature')
    plt.ylabel('⟨E⟩')
    plt.subplot(1,2,2)
    plt.plot(temperatures, avg_magnetizations, 'o-')
    plt.xlabel('Temperature')
    plt.ylabel('⟨|M|⟩')
    plt.suptitle(f'Phase Transition (L={L})')
    plt.show()

# Part c: Magnetization vs time for L=5, T=2.2
def run_magnetization_vs_time(T=2.2, L=5):
    accept_prob = precompute_accept_prob(T)
    spins = initialize_lattice(L)
    
    # Thermalization (N_therm sweeps)
    for _ in range(N_therm):
        for _ in range(L * L):
            i, j = np.random.randint(0, L, 2)
            S = spins[i, j]
            neighbors = spins[(i+1)%L, j] + spins[i, (j+1)%L] + spins[(i-1)%L, j] + spins[i, (j-1)%L]
            dE = 2 * J * S * neighbors
            prob = accept_prob.get(dE, np.exp(-dE/(k_B*T)) if dE > 0 else 1.0)
            if np.random.rand() < prob:
                spins[i, j] *= -1
    
    # Measurement (N_sample sweeps)
    magnetization = []
    for _ in range(N_sample):
        for _ in range(L * L):
            i, j = np.random.randint(0, L, 2)
            S = spins[i, j]
            neighbors = spins[(i+1)%L, j] + spins[i, (j+1)%L] + spins[(i-1)%L, j] + spins[i, (j-1)%L]
            dE = 2 * J * S * neighbors
            prob = accept_prob.get(dE, np.exp(-dE/(k_B*T)) if dE > 0 else 1.0)
            if np.random.rand() < prob:
                spins[i, j] *= -1
        magnetization.append(compute_magnetization(spins))  # No absolute value
    
    return magnetization

# Run simulation and plot
magnetization = run_magnetization_vs_time()
plt.figure(figsize=(8, 5))
plt.plot(magnetization, alpha=0.8)
plt.xlabel("Simulation Time Step")
plt.ylabel("Magnetization $M$")
plt.title("Magnetization vs. Time (L=5, T=2.2)")
plt.show()