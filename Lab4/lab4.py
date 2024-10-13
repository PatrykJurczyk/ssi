import random
import matplotlib.pyplot as plt
import numpy as np

def fitness_function(x):
    return np.sin(x / 10) * np.sin(x / 200)

def one_plus_one_algorithm(dispersion, growth_factor, num_iterations, value_range=(0, 100)):
    x = random.uniform(value_range[0], value_range[1])
    y = fitness_function(x)
    iterations = []
    x_values = []
    y_values = []
    
    x_plot_values = np.linspace(value_range[0], value_range[1], 1000)
    y_plot_values = fitness_function(x_plot_values)

    plt.plot(x_plot_values, y_plot_values, label="Fitness Function")
    
    for i in range(num_iterations):
        x_candidate = x + random.uniform(-dispersion, dispersion)

        x_candidate = max(min(x_candidate, value_range[1]), value_range[0])
        
        y_candidate = fitness_function(x_candidate)
        
        print(f"Iteration: {i+1}, x: {x:.4f}, y: {y:.4f}, dispersion: {dispersion:.4f}")
        
        if y_candidate >= y:
            x, y = x_candidate, y_candidate
            dispersion *= growth_factor
        else:
            dispersion /= growth_factor

        iterations.append(i + 1)
        x_values.append(x)
        y_values.append(y)

        plt.scatter(x, y, color='blue')

    plt.scatter(x, y, color='red', label='Best Result', zorder=5)
    plt.title("1+1 Algorithm - Maximizing the Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return x, y

initial_dispersion = 10
growth_factor = 1.1
num_iterations = 100
value_range = (0, 100)

best_x, best_y = one_plus_one_algorithm(initial_dispersion, growth_factor, num_iterations, value_range)

print(f"Best found point: x = {best_x:.4f}, y = {best_y:.4f}")
