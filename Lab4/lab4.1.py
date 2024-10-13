import numpy as np
import random
import matplotlib.pyplot as plt

def fitness_function(x1, x2):
    return np.sin(x1 * 0.05) + np.sin(x2 * 0.05) + 0.4 * np.sin(x1 * 0.15) * np.sin(x2 * 0.15)

def initialize_population(population_size, value_range):
    return np.random.uniform(value_range[0], value_range[1], (population_size, 2))

def tournament_selection(population, tournament_size):
    tournament = random.sample(list(population), tournament_size)
    best_individual = max(tournament, key=lambda ind: fitness_function(ind[0], ind[1]))
    return best_individual

def crossover(parent1, parent2):
    crossover_point = random.randint(1, 2)
    if crossover_point == 1:
        child1 = np.array([parent1[0], parent2[1]])
        child2 = np.array([parent2[0], parent1[1]])
    else:
        child1 = np.array([parent1[0], parent2[1]])
        child2 = np.array([parent2[0], parent1[1]])
    return child1, child2

def mutate(individual, mutation_rate, value_range):
    if random.random() < 0.1:
        individual[0] += np.random.uniform(-mutation_rate, mutation_rate)
        individual[1] += np.random.uniform(-mutation_rate, mutation_rate)
        individual = np.clip(individual, value_range[0], value_range[1])
    return individual

def evolutionary_algorithm(population_size, offspring_size, tournament_size, mutation_rate, num_iterations, value_range):
    population = initialize_population(population_size, value_range)
    best_individuals = []

    for iteration in range(num_iterations):
        new_individuals = []
        
        for _ in range(offspring_size // 2):
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            new_individuals.append(mutate(child1, mutation_rate, value_range))
            new_individuals.append(mutate(child2, mutation_rate, value_range))
        
        population = np.vstack((population, new_individuals))
        population = sorted(population, key=lambda ind: fitness_function(ind[0], ind[1]), reverse=True)[:population_size]

        best_individuals.append(population[0])

        print(f"Iteration {iteration + 1}: Best individual: {population[0]}, Fitness value: {fitness_function(population[0][0], population[0][1])}")

        if iteration % 5 == 0 or iteration == num_iterations - 1:
            visualize(population, best_individuals, value_range, iteration)

    best_overall_individual = max(population, key=lambda ind: fitness_function(ind[0], ind[1]))
    return best_overall_individual, best_individuals

def visualize(population, best_individuals, value_range, iteration):
    """Wizualizuje populację oraz najlepszego osobnika w danej iteracji."""
    x1 = np.linspace(value_range[0], value_range[1], 400)
    x2 = np.linspace(value_range[0], value_range[1], 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = fitness_function(X1, X2)

    plt.figure(figsize=(10, 8))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Fitness value')
    
    population_np = np.array(population)
    plt.scatter(population_np[:, 0], population_np[:, 1], c='red', marker='o', s=100, label="Individuals")
    
    best_individual = best_individuals[-1]
    plt.scatter(best_individual[0], best_individual[1], c='blue', marker='x', s=200, label="Best individual")
    
    plt.title(f'Iteration {iteration + 1}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

population_size = 4
offspring_size = 10
tournament_size = 2
mutation_rate = 10
num_iterations = 20
value_range = (0, 100)

best_individual, best_individuals = evolutionary_algorithm(population_size, offspring_size, tournament_size, mutation_rate, num_iterations, value_range)

x1_results = [ind[0] for ind in best_individuals]
x2_results = [ind[1] for ind in best_individuals]
y_results = [fitness_function(ind[0], ind[1]) for ind in best_individuals]

plt.plot(range(1, num_iterations + 1), y_results, marker='o')
plt.title('Best results over iterations')
plt.xlabel('Iteration')
plt.ylabel('Fitness value')
plt.xticks(range(1, num_iterations + 1))
plt.grid()
plt.show()

best_fitness_value = fitness_function(best_individual[0], best_individual[1])
print(f"\nBest individual: {best_individual}, Fitness value: {best_fitness_value}")
