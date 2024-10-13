import numpy as np
import random
import matplotlib.pyplot as plt

def fitness_function(x1, x2):
    return np.sin(0.05 * x1) + np.sin(0.05 * x2) + 0.4 * np.sin(0.15 * x1) * np.sin(0.15 * x2)

def pso(num_particles, inertia_weight, global_weight, local_weight, num_iterations, xmin, xmax):
    population = np.random.uniform(xmin, xmax, (num_particles, 2))
    velocities = np.zeros((num_particles, 2))
    local_best = np.copy(population)
    local_best_scores = np.array([fitness_function(ind[0], ind[1]) for ind in local_best])
    
    global_best_score = np.max(local_best_scores)
    global_best = local_best[np.argmax(local_best_scores)]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    x1_vals = np.linspace(xmin, xmax, 100)
    x2_vals = np.linspace(xmin, xmax, 100)
    x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
    z_vals = fitness_function(x1_grid, x2_grid)

    ax.plot_surface(x1_grid, x2_grid, z_vals, cmap='viridis', alpha=0.6)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('F(x1, x2)')

    for t in range(num_iterations):
        for j in range(num_particles):
            current_score = fitness_function(population[j][0], population[j][1])

            if current_score > local_best_scores[j]:
                local_best_scores[j] = current_score
                local_best[j] = population[j]

        best_particle = local_best[np.argmax(local_best_scores)]
        global_best_score = np.max(local_best_scores)

        for j in range(num_particles):
            rnd_global = np.random.rand()
            rnd_local = np.random.rand()

            for i in range(2):
                velocities[j, i] = (
                    velocities[j, i] * inertia_weight
                    + global_weight * rnd_global * (best_particle[i] - population[j, i])
                    + local_weight * rnd_local * (local_best[j, i] - population[j, i])
                )
                population[j, i] += velocities[j, i]
            population[j] = np.clip(population[j], xmin, xmax)

        ax.scatter(population[:, 0], population[:, 1], [fitness_function(x[0], x[1]) for x in population], color='red', label=f'Iteration {t + 1}' if t == 0 else "")
        plt.pause(0.1)

        print(f"Iteration {t + 1}: Global best x1 = {best_particle[0]:.4f}, x2 = {best_particle[1]:.4f}, Score: {global_best_score:.4f}")

    plt.show()

    return best_particle, global_best_score

N = 5
inertia_weight = 0.2
global_weight = 0.3
local_weight = 0.3
num_iterations = 100
xmin, xmax = 0, 100

best_particle, best_score = pso(N, inertia_weight, global_weight, local_weight, num_iterations, xmin, xmax)

print(f"\nBest solution: x1 = {best_particle[0]:.4f}, x2 = {best_particle[1]:.4f}, Score: {best_score:.4f}")
