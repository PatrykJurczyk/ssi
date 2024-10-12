import random
import matplotlib.pyplot as plt
import numpy as np

class Fireflies:
    def __init__(self, N, beta_zero, gamma_zero, mu_zero, x_min_i=0, x_max_i=100, iterations=30):
        self.N = N
        self.beta_zero = beta_zero
        self.gamma = gamma_zero / (x_max_i - x_min_i)
        self.mu_i = (x_max_i - x_min_i) * mu_zero
        self.iterations = iterations
        self.X = np.random.uniform(x_min_i, x_max_i, (N, 2))
        self.F = np.array([self.evaluate_function(x) for x in self.X])
        self.best_point = [self.X[np.argmax(self.F)], np.max(self.F)]

    def evaluate_function(self, position):
        x1, x2 = position
        return np.sin(x1 * 0.05) + np.sin(x2 * 0.05) + 0.4 * np.sin(x1 * 0.15) * np.sin(x2 * 0.15)

    def update_best_point(self, index):
        if self.F[index] > self.best_point[1]:
            self.best_point = [self.X[index], self.F[index]]

    def visualize(self, iteration):
        if iteration == 0:
            iteration = 'INITIAL'
        x_1, x_2 = np.meshgrid(np.linspace(-25, 125, 400), np.linspace(-25, 125, 400))
        z = self.evaluate_function([x_1, x_2])
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(x_1, x_2, z, levels=50, cmap='viridis')
        plt.colorbar(contour)
        plt.scatter(self.X[:, 0], self.X[:, 1], c='red', marker='o', s=100, label="Fireflies")
        plt.scatter(*self.best_point[0], c='blue', marker='x', s=200, label="Best Point")
        plt.title(f'Fireflies iteration {iteration}')
        plt.legend()
        plt.show()

    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    def move_fireflies(self):
        for i in random.sample(range(self.N), self.N):
            for j in random.sample(range(self.N), self.N):
                if self.F[j] > self.F[i]:
                    distance = self.euclidean_distance(self.X[i], self.X[j])
                    beta = self.beta_zero * np.exp(-self.gamma * distance ** 2)
                    self.X[i] += beta * (self.X[j] - self.X[i])
            self.X[i] += np.random.uniform(-self.mu_i, self.mu_i, 2)
            self.F[i] = self.evaluate_function(self.X[i])
            self.update_best_point(i)

    def optimize(self, visualize=False, visualize_interval=1):
        for i in range(self.iterations):
            self.move_fireflies()
            if visualize and (i % visualize_interval == 0 or i == self.iterations - 1):
                self.visualize(i)

    def get_best_point(self):
        return self.best_point

def run_fireflies_algorithm():
    ff = Fireflies(4, 0.3, 0.1, 0.05, iterations=100)
    ff.optimize(visualize=True, visualize_interval=20)
    print("Najlepszy punkt (x1, x2):\n", ff.get_best_point()[0])

def main():
    run_fireflies_algorithm()

if __name__ == '__main__':
    main()
