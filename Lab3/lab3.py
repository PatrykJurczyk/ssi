import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lab2.lab2 import draw_points

class Dataset:
    def __init__(self):
        self.dataframe = pd.DataFrame()

    def load_data(self, filename: str):
        self.dataframe = pd.read_csv(filename, sep='\s+', header=None)

    def convert_samples_to_numbers(self, attribute_indices: list = None):
        if not all(len(row) == len(self.dataframe.iloc[0]) for row in self.dataframe.values):
            raise ValueError("Rows have different lengths.")

        try:
            if attribute_indices is not None:
                self.dataframe = self.dataframe.iloc[:, attribute_indices].apply(pd.to_numeric, errors='coerce')
            else:
                self.dataframe = self.dataframe.apply(pd.to_numeric, errors='coerce')
            self.dataframe.dropna(inplace=True)
        except Exception as e:
            print(f"Error during conversion: {e}")

    def get_data_as_array(self) -> np.ndarray:
        return self.dataframe.to_numpy()

class KMeansClustering:
    def __init__(self, data: np.ndarray, k: int, max_iterations: int = 100):
        self.data = data
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.membership = None

    def initialize_centroids(self):
        if self.data.size == 0:
            raise ValueError("Cannot initialize centroids because data is empty.")
        
        initial_indices = np.random.choice(len(self.data), self.k, replace=False)
        self.centroids = self.data[initial_indices]

    def visualize_clusters(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        if self.data.shape[1] < 2:
            print("Error: Too few dimensions for visualization.")
            return

        for cluster_index in range(self.k):
            cluster_data = self.data[self.membership == cluster_index]
            if cluster_data.size > 0:
                draw_points(ax, cluster_data[:, 0], cluster_data[:, 1], cluster_index)

        if self.centroids is not None:
            draw_points(ax, self.centroids[:, 0], self.centroids[:, 1], self.k)
        
        ax.set_title('K-Means Clustering')
        ax.legend([f'Group {i + 1}' for i in range(self.k)] + ['Centroids'])
        plt.show()

    def fit(self):
        self.initialize_centroids()
        for iteration in range(self.max_iterations):
            distances = np.linalg.norm(self.data[:, np.newaxis] - self.centroids, axis=2)
            self.membership = np.argmin(distances, axis=1)

            new_centroids = np.array([self.data[self.membership == k].mean(axis=0) for k in range(self.k)])
            if np.any(np.isnan(new_centroids)):
                print("Error: Failed to create all centroids.")
                return
            if np.all(self.centroids == new_centroids):
                print(f"Algorithm converged after {iteration} iterations.")
                break

            self.centroids = new_centroids

            print(f"Iteration {iteration + 1}:")
            print("Centroids:", self.centroids)
            print("Membership:", self.membership)

    def get_centroids(self) -> np.ndarray:
        return self.centroids

    def get_membership(self) -> np.ndarray:
        return self.membership

def zad1 ():
    sample_strings = [["1", "a", "2.2"], ["3", "4", "5"]]
    attribute_indices = [0, 2]

    dataset = Dataset()
    dataset.dataframe = pd.DataFrame(sample_strings)
    dataset.convert_samples_to_numbers(attribute_indices)
    print("Data after conversion:", dataset.get_data_as_array())

def zad2 ():
    dataset = Dataset()
    dataset.load_data('spiralka.txt')
    dataset.convert_samples_to_numbers()

    if dataset.get_data_as_array().size == 0:
        print("Error: Loaded data is empty.")
        return

    kmeans = KMeansClustering(dataset.get_data_as_array(), k=4)
    kmeans.fit()
    
    if kmeans.centroids is not None:
        kmeans.visualize_clusters()
        print("Final centroid points:\n", kmeans.get_centroids())

def main():
    task1()
    task2()

if __name__ == '__main__':
    np.random.seed(0)
    main()
