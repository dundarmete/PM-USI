import numpy as np
import matplotlib.pyplot as plt
import random

# Simulation Parameters (from homework description and part a)
L = 10.0  # Habitat size
N_r_initial = 900  # Initial number of rabbits
N_w_initial = 100  # Initial number of wolves
sigma = 0.5  # Step size standard deviation for both
p_r_rabbit = 0.02  # Rabbit replication probability
t_d_rabbit = 100  # Rabbit death age (for part a)
r_c = 0.5  # Wolf eating radius
p_e_wolf = 0.02  # Wolf eating probability per rabbit in range
p_r_wolf = 0.02  # Wolf replication probability per eaten rabbit
t_d_wolf = 50  # Wolf death time without food

# Simulation runtime
n_steps = 3000 # Recommended a few thousand steps

# --- Agent Class Definitions ---

class Agent:
    """Base class for agents (rabbits and wolves)."""
    def __init__(self, x, y, L):
        self.x = x
        self.y = y
        self.L = L

    def move(self, sigma):
        """Moves the agent with periodic boundary conditions."""
        # Choose random direction
        angle = random.uniform(0, 2 * np.pi)
        # Sample step length from Normal(0, sigma) - use absolute value or redraw if negative?
        # The description implies length, suggesting non-negative. Let's use absolute value.
        # A true Normal distribution allows negative values, interpreting as step *magnitude*
        # Let's sample dx, dy from Normal(0, sigma/sqrt(2)) for isotropic steps
        # Or sample angle and magnitude separately. Let's stick to angle + magnitude.
        # Using np.random.normal for magnitude directly. Taking abs value as length.
        step_length = abs(np.random.normal(0, sigma))

        dx = step_length * np.cos(angle)
        dy = step_length * np.sin(angle)

        self.x = (self.x + dx) % self.L
        self.y = (self.y + dy) % self.L

    def get_pos(self):
        return self.x, self.y

class Rabbit(Agent):
    """Represents a rabbit."""
    def __init__(self, x, y, L, max_age):
        super().__init__(x, y, L)
        # Initial age uniformly sampled from [1, max_age)
        self.age = random.randint(1, max_age - 1)
        self.max_age = max_age

    def step(self, sigma, p_replicate):
        """Rabbit behavior for one time step."""
        self.move(sigma)
        self.age += 1
        # Check for replication
        replicated = False
        if random.random() < p_replicate:
            replicated = True
        # Check for death by old age
        is_alive = self.age < self.max_age
        return is_alive, replicated

class Wolf(Agent):
    """Represents a wolf."""
    def __init__(self, x, y, L, starve_time):
        super().__init__(x, y, L)
        self.starve_time = starve_time
        self.time_since_last_meal = 0

    def step(self, sigma):
        """Wolf movement for one time step."""
        self.move(sigma)
        self.time_since_last_meal += 1

    def check_starvation(self):
        """Checks if the wolf dies from hunger."""
        # Dies if it hasn't eaten for t_d_wolf steps
        return self.time_since_last_meal < self.starve_time

    def attempt_eat(self, rabbits, p_eat, r_capture, p_replicate):
        """Attempts to eat nearby rabbits and replicate."""
        eaten_count = 0
        replicated_count = 0
        indices_to_remove = []
        wolf_x, wolf_y = self.get_pos()

        for i, rabbit in enumerate(rabbits):
            rabbit_x, rabbit_y = rabbit.get_pos()
            # Calculate distance considering periodic boundaries
            dx = abs(wolf_x - rabbit_x)
            dy = abs(wolf_y - rabbit_y)
            dist_x = min(dx, self.L - dx)
            dist_y = min(dy, self.L - dy)
            distance = np.sqrt(dist_x**2 + dist_y**2)

            if distance < r_capture:
                # Attempt to eat with probability p_eat
                if random.random() < p_eat:
                    indices_to_remove.append(i)
                    eaten_count += 1
                    self.time_since_last_meal = 0 # Reset hunger clock
                    # Attempt to replicate with probability p_replicate
                    if random.random() < p_replicate:
                        replicated_count += 1

        # Return counts and indices of eaten rabbits (in reverse order for safe removal)
        return eaten_count, replicated_count, sorted(indices_to_remove, reverse=True)

# --- Initialization ---

rabbits = []
wolves = []

# Randomly distribute initial rabbits
for _ in range(N_r_initial):
    x = random.uniform(0, L)
    y = random.uniform(0, L)
    rabbits.append(Rabbit(x, y, L, t_d_rabbit))

# Randomly distribute initial wolves
for _ in range(N_w_initial):
    x = random.uniform(0, L)
    y = random.uniform(0, L)
    wolves.append(Wolf(x, y, L, t_d_wolf))

# Data logging
time_points = []
rabbit_counts = []
wolf_counts = []

# --- Simulation Loop ---

print(f"Starting simulation: {N_r_initial} rabbits, {N_w_initial} wolves, L={L}, sigma={sigma}, t_d_rabbit={t_d_rabbit}")

for t in range(n_steps):
    # 0. Log current state
    time_points.append(t)
    rabbit_counts.append(len(rabbits))
    wolf_counts.append(len(wolves))

    if t % 100 == 0: # Print progress
       print(f"Step {t}: Rabbits={len(rabbits)}, Wolves={len(wolves)}")
    if not rabbits or not wolves: # Stop if one population dies out
        print(f"Simulation stopped at step {t} due to extinction.")
        break


    # 1. Wolf actions (eating and replication)
    new_wolves = []
    total_eaten_this_step = 0
    eaten_indices_global = set() # Keep track of globally eaten rabbits this step

    next_wolves = []
    for i, wolf in enumerate(wolves):
        if not wolf.check_starvation(): # Skip starved wolves (they die later)
             continue

        eaten_count, rep_count, eaten_indices_local = wolf.attempt_eat(
            rabbits, p_e_wolf, r_c, p_r_wolf
        )

        if eaten_count > 0:
            # Add indices relative to the current rabbit list *before* any removals
            for local_idx in eaten_indices_local:
                 # Ensure we don't process an index already marked for removal globally
                 if local_idx not in eaten_indices_global:
                    eaten_indices_global.add(local_idx)

        # Wolf attempts replication for *each* rabbit eaten
        for _ in range(eaten_count):
             if random.random() < p_r_wolf:
                 # Create new wolf at the parent's location
                 wx, wy = wolf.get_pos()
                 new_wolves.append(Wolf(wx, wy, L, t_d_wolf))


    # Remove eaten rabbits *after* all wolves have attempted eating
    # Convert set to sorted list (descending) for safe removal
    sorted_eaten_indices = sorted(list(eaten_indices_global), reverse=True)
    if sorted_eaten_indices:
        # print(f"Step {t}: Removing {len(sorted_eaten_indices)} rabbits at indices: {sorted_eaten_indices}")
        pass
    surviving_rabbits_after_predation = []
    current_global_index = 0
    eaten_set_for_iteration = set(sorted_eaten_indices) # Use a set for quick lookups
    for i, r in enumerate(rabbits):
        if i not in eaten_set_for_iteration:
            surviving_rabbits_after_predation.append(r)

    rabbits = surviving_rabbits_after_predation
    total_eaten_this_step = len(sorted_eaten_indices)


    # 2. Rabbit actions (movement, replication, death)
    next_rabbits = []
    new_rabbits = []
    for rabbit in rabbits:
        is_alive, replicated = rabbit.step(sigma, p_r_rabbit)
        if is_alive:
            next_rabbits.append(rabbit)
            if replicated:
                # Create new rabbit at the same position
                rx, ry = rabbit.get_pos()
                # New rabbit starts with age 0 (or 1?) - Let's assume age 0/1
                new_rabbits.append(Rabbit(rx, ry, L, t_d_rabbit)) # Give it age 1? Needs clarification, using 1 as per initial age logic
                new_rabbits[-1].age = 1 # Set age explicitly if needed


    # 3. Wolf actions (movement, starvation death)
    next_wolves = []
    for wolf in wolves:
        wolf.step(sigma) # Move and increment hunger timer
        if wolf.check_starvation(): # Check if wolf survives
            next_wolves.append(wolf)

    # 4. Update populations
    rabbits = next_rabbits + new_rabbits
    wolves = next_wolves + new_wolves


# Final counts
time_points.append(n_steps)
rabbit_counts.append(len(rabbits))
wolf_counts.append(len(wolves))
print(f"Simulation finished at step {n_steps}: Rabbits={len(rabbits)}, Wolves={len(wolves)}")

# --- Plotting Results ---
plt.figure(figsize=(10, 6))
plt.plot(time_points, rabbit_counts, label='Rabbits')
plt.plot(time_points, wolf_counts, label='Wolves')
plt.xlabel('Time Steps')
plt.ylabel('Population Size')
plt.title('Prey-Predator Simulation (Part a parameters)')
plt.legend()
plt.grid(True)
plt.show()