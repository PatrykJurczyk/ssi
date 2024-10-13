import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.n = width * height
        self.weights = np.zeros((self.n, self.n))

    def train(self, pattern):
        reshaped_pattern = pattern.flatten()
        reshaped_pattern = np.where(reshaped_pattern == 0, -1, reshaped_pattern)
        self.weights += np.outer(reshaped_pattern, reshaped_pattern)
        np.fill_diagonal(self.weights, 0)

    def denoise(self, noisy_pattern):
        flat_pattern = noisy_pattern.flatten()
        flat_pattern = np.where(flat_pattern == 0, -1, flat_pattern)
        for _ in range(10):
            for i in range(self.n):
                sum_input = 0
                for j in range(self.n):
                    if i != j:
                        sum_input += self.weights[i, j] * flat_pattern[j]
                flat_pattern[i] = 1 if sum_input >= 0 else -1
        return flat_pattern.reshape(self.height, self.width)

    def display(self, pattern):
        plt.imshow(pattern, cmap='gray', vmin=-1, vmax=1)
        plt.title('Repaired Image')
        plt.axis('off')
        plt.show()

def main():
    width, height = 5, 5
    network = HopfieldNetwork(width, height)

    patterns = [
        np.array([[1, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0]]),
        np.array([[1, 0, 0, 0, 1],
                   [0, 1, 0, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, 0, 1, 0],
                   [1, 0, 0, 0, 1]]),
        np.array([[0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0]]),
    ]

    test_patterns = [
        np.array([[0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0]]),
        np.array([[1, 1, 0, 0, 1],
                   [0, 1, 0, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 0, 1, 0],
                   [1, 1, 0, 0, 1]]),
        np.array([[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0]]),
        np.array([[0, 1, 1, 1, 1],
                   [1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1]]),
    ]

    for pattern in patterns:
        network.train(pattern)

    for test_pattern in test_patterns:
        repaired_pattern = network.denoise(test_pattern)
        network.display(repaired_pattern)

if __name__ == '__main__':
    main()
