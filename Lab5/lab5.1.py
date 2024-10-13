import numpy as np
import matplotlib.pyplot as plt

def fitness_function(x1, x2):
    return np.sin(0.05 * x1) + np.sin(0.05 * x2) + 0.4 * np.sin(0.15 * x1) * np.sin(0.15 * x2)

def euclidean_distance(firefly_a, firefly_b):
    return np.sqrt(np.sum((firefly_a - firefly_b) ** 2))

def firefly_algorithm(num_fireflies, initial_beta, initial_gamma, mutation_rate, num_iterations, xmin, xmax):
    population = np.random.uniform(xmin, xmax, (num_fireflies, 2))
    evaluations = np.array([fitness_function(ind[0], ind[1]) for ind in population])
    
    max_distance = np.sqrt(2 * (xmax - xmin) ** 2)
    gamma = initial_gamma / max_distance

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

    for iteration in range(num_iterations):
        for a in range(num_fireflies):
            for b in range(num_fireflies):
                if evaluations[b] > evaluations[a]:
                    distance_ab = euclidean_distance(population[a], population[b])
                    beta = initial_beta * np.exp(-gamma * distance_ab ** 2)
                    population[a] += beta * (population[b] - population[a])
                    mutation = mutation_rate * (xmax - xmin) * (np.random.rand(2) - 0.5)
                    population[a] += mutation

                    population[a] = np.clip(population[a], xmin, xmax)

            evaluations[a] = fitness_function(population[a][0], population[a][1])

        ax.scatter(population[:, 0], population[:, 1], evaluations, color='red', label=f'Iteration {iteration + 1}' if iteration == 0 else "")
        plt.pause(0.1)

        best_index = np.argmax(evaluations)
        print(f"Iteration {iteration + 1}: Best x1 = {population[best_index][0]:.4f}, x2 = {population[best_index][1]:.4f}, Evaluation: {evaluations[best_index]:.4f}")

    plt.show()

    best_index = np.argmax(evaluations)
    return population[best_index], evaluations[best_index]

num_fireflies = 5
initial_beta = 0.3
initial_gamma = 0.1
mutation_rate = 0.05
num_iterations = 30
xmin, xmax = 0, 100

best_firefly, best_evaluation = firefly_algorithm(num_fireflies, initial_beta, initial_gamma, mutation_rate, num_iterations, xmin, xmax)

print(f"\nBest solution: x1 = {best_firefly[0]:.4f}, x2 = {best_firefly[1]:.4f}, Evaluation: {best_evaluation:.4f}")
