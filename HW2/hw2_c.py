import numpy as np
import matplotlib.pyplot as plt
import random

# Simulation Parameters (modified for part c)
L = 8.0  # Habitat size (Can be changed to 8.0 if too slow)
N_r_initial = 900  # Initial number of rabbits
N_w_initial = 100  # Initial number of wolves
# --- MODIFICATION FOR PART (c) ---
sigma = 0.05  # Step size standard deviation for both (changed from 0.5)
# ---------------------------------
p_r_rabbit = 0.02  # Rabbit replication probability
# --- MODIFICATION FOR PART (c) ---
t_d_rabbit = 100  # Rabbit death age (back to part a value)
# ---------------------------------
r_c = 0.5  # Wolf eating radius
p_e_wolf = 0.02  # Wolf eating probability per rabbit in range
p_r_wolf = 0.02  # Wolf replication probability per eaten rabbit
t_d_wolf = 50  # Wolf death time without food

# Simulation runtime
n_steps = 3000 # Recommended a few thousand steps (might need more for slow dynamics)

# --- Agent Class Definitions --- (Identical to previous code)

class Agent:
    """Base class for agents (rabbits and wolves)."""
    def __init__(self, x, y, L):
        self.x = x
        self.y = y
        self.L = L

    def move(self, sigma):
        """Moves the agent with periodic boundary conditions."""
        angle = random.uniform(0, 2 * np.pi)
        # Step length sampled from Normal(0, sigma)
        # Using abs() as step length cannot be negative
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
        self.age = random.randint(1, max(1, max_age - 1))
        self.max_age = max_age # Should be 100 again

    def step(self, sigma, p_replicate):
        """Rabbit behavior for one time step."""
        self.move(sigma) # Uses the new sigma=0.05
        self.age += 1
        replicated = False
        if random.random() < p_replicate:
            replicated = True
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
        self.move(sigma) # Uses the new sigma=0.05
        self.time_since_last_meal += 1

    def check_starvation(self):
        """Checks if the wolf dies from hunger."""
        return self.time_since_last_meal < self.starve_time

    def attempt_eat(self, rabbits, p_eat, r_capture, p_replicate):
        """Attempts to eat nearby rabbits and replicate."""
        eaten_count = 0
        replicated_count = 0
        indices_to_remove = []
        wolf_x, wolf_y = self.get_pos()

        for i, rabbit in enumerate(rabbits):
            rabbit_x, rabbit_y = rabbit.get_pos()
            dx = abs(wolf_x - rabbit_x)
            dy = abs(wolf_y - rabbit_y)
            dist_x = min(dx, self.L - dx)
            dist_y = min(dy, self.L - dy)
            distance = np.sqrt(dist_x**2 + dist_y**2)

            if distance < r_capture:
                if random.random() < p_eat:
                    indices_to_remove.append(i)
                    eaten_count += 1
                    self.time_since_last_meal = 0
                    if random.random() < p_replicate:
                         replicated_count += 1

        return eaten_count, replicated_count, sorted(indices_to_remove, reverse=True)


# --- Initialization --- (Uses part a/c parameters)

rabbits = []
wolves = []

# Initialize rabbits with t_d_rabbit = 100
for _ in range(N_r_initial):
    x = random.uniform(0, L)
    y = random.uniform(0, L)
    rabbits.append(Rabbit(x, y, L, t_d_rabbit)) # t_d_rabbit is 100

# Initialize wolves
for _ in range(N_w_initial):
    x = random.uniform(0, L)
    y = random.uniform(0, L)
    wolves.append(Wolf(x, y, L, t_d_wolf))

# Data logging
time_points = []
rabbit_counts = []
wolf_counts = []

# --- Simulation Loop --- (Identical logic)

print(f"Starting simulation: {N_r_initial} rabbits, {N_w_initial} wolves, L={L}, sigma={sigma}, t_d_rabbit={t_d_rabbit}") # Will show sigma=0.05, t_d_rabbit=100

for t in range(n_steps):
    time_points.append(t)
    rabbit_counts.append(len(rabbits))
    wolf_counts.append(len(wolves))

    if t % 100 == 0:
       print(f"Step {t}: Rabbits={len(rabbits)}, Wolves={len(wolves)}")
    # Consider adding a check for excessively large populations if needed
    if not rabbits or not wolves:
        print(f"Simulation stopped at step {t} due to extinction.")
        break

    # 1. Wolf actions (eating and replication)
    new_wolves_from_replication = []
    eaten_indices_global = set()

    for i, wolf in enumerate(wolves):
         if not wolf.check_starvation():
             continue
         eaten_count, rep_count, eaten_indices_local = wolf.attempt_eat(
            rabbits, p_e_wolf, r_c, p_r_wolf
         )
         for local_idx in eaten_indices_local:
             if local_idx not in eaten_indices_global:
                eaten_indices_global.add(local_idx)
         for _ in range(rep_count):
             wx, wy = wolf.get_pos()
             new_wolves_from_replication.append(Wolf(wx, wy, L, t_d_wolf))


    # Remove eaten rabbits globally
    sorted_eaten_indices = sorted(list(eaten_indices_global), reverse=True)
    surviving_rabbits_after_predation = []
    eaten_set_for_iteration = set(sorted_eaten_indices)
    for i, r in enumerate(rabbits):
        if i not in eaten_set_for_iteration:
            surviving_rabbits_after_predation.append(r)
    rabbits = surviving_rabbits_after_predation

    # 2. Rabbit actions (movement, replication, death by age)
    next_rabbits = []
    new_rabbits_from_replication = []
    for rabbit in rabbits:
        # Pass the new sigma value here
        is_alive, replicated = rabbit.step(sigma, p_r_rabbit)
        if is_alive:
            next_rabbits.append(rabbit)
            if replicated:
                rx, ry = rabbit.get_pos()
                new_rabbit = Rabbit(rx, ry, L, t_d_rabbit)
                new_rabbit.age = 1
                new_rabbits_from_replication.append(new_rabbit)

    # 3. Wolf actions (movement, check starvation death)
    next_wolves = []
    for wolf in wolves:
        # Pass the new sigma value here
        wolf.step(sigma)
        if wolf.check_starvation():
            next_wolves.append(wolf)

    # 4. Update populations for the next step
    rabbits = next_rabbits + new_rabbits_from_replication
    wolves = next_wolves + new_wolves_from_replication


# Final counts
final_step = t if t < n_steps else n_steps # Use actual stop step if extinction occurred
time_points.append(final_step)
rabbit_counts.append(len(rabbits))
wolf_counts.append(len(wolves))
print(f"Simulation finished at step {final_step}: Rabbits={len(rabbits)}, Wolves={len(wolves)}")


# --- Plotting Results ---
plt.figure(figsize=(10, 6))
plt.plot(time_points, rabbit_counts, label='Rabbits')
plt.plot(time_points, wolf_counts, label='Wolves')
plt.xlabel('Time Steps')
plt.ylabel('Population Size')
plt.title(f'Prey-Predator Simulation (Part c: sigma={sigma})') # Updated title
plt.legend()
plt.grid(True)
plt.show()