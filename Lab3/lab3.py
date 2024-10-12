import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Lab2.lab2 import draw_points

class Data:
    def __init__(self):
        self.data = pd.DataFrame()

    def read_data(self, nazwa_pliku_z_wartosciami: str):
        self.data = pd.read_csv(nazwa_pliku_z_wartosciami, sep='\s+', header=None)

    def probki_str_na_liczby(self, numery_atr: list = None):
        if not all(len(row) == len(self.data.iloc[0]) for row in self.data.values):
            raise ValueError("Wiersze mają różne długości.")

        try:
            if numery_atr is not None:
                self.data = self.data.iloc[:, numery_atr].apply(pd.to_numeric, errors='coerce')
            else:
                self.data = self.data.apply(pd.to_numeric, errors='coerce')
            self.data.dropna(inplace=True)
        except Exception as e:
            print(f"Błąd podczas konwersji: {e}")

    def get_data(self) -> np.ndarray:
        """Zwraca dane jako macierz NumPy."""
        return self.data.to_numpy()

class KMeans:
    def __init__(self, data: np.ndarray, k: int, max_iterations: int = 100):
        self.data = data
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.membership = None

    def initialize_centroids(self):
        if self.data.size == 0:
            raise ValueError("Nie można zainicjalizować centroidów, ponieważ dane są puste.")
        
        initial_indices = np.random.choice(len(self.data), self.k, replace=False)
        self.centroids = self.data[initial_indices]

    def visualize(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        if self.data.shape[1] < 2:
            print("Błąd: Zbyt mało wymiarów do wizualizacji.")
            return

        for x in range(self.k):
            cluster_data = self.data[self.membership == x]
            if cluster_data.size > 0:
                draw_points(ax, cluster_data[:, 0], cluster_data[:, 1], x)

        if self.centroids is not None:
            draw_points(ax, self.centroids[:, 0], self.centroids[:, 1], self.k)
        
        ax.set_title('Klasteryzacja K-średnich')
        ax.legend([f'Grupa {i + 1}' for i in range(self.k)] + ['Centroidy'])
        plt.show()

    def fit(self):
        self.initialize_centroids()
        for iteration in range(self.max_iterations):
            distances = np.linalg.norm(self.data[:, np.newaxis] - self.centroids, axis=2)
            self.membership = np.argmin(distances, axis=1)

            new_centroids = np.array([self.data[self.membership == k].mean(axis=0) for k in range(self.k)])
            if np.any(np.isnan(new_centroids)):
                print("Błąd: Nie udało się utworzyć wszystkich centroidów.")
                return
            if np.all(self.centroids == new_centroids):
                print(f"Algorytm zbiega po {iteration} iteracjach.")
                break

            self.centroids = new_centroids

            print(f"Iteracja {iteration + 1}:")
            print("Centroidy:", self.centroids)
            print("Przynależność:", self.membership)

    def get_centroids(self) -> np.ndarray:
        return self.centroids

    def get_membership(self) -> np.ndarray:
        return self.membership

def zad1():
    probki_str = [["1", "a", "2.2"], ["3", "4", "5"]]
    numery_atr = [0, 2]

    data = Data()
    data.data = pd.DataFrame(probki_str)
    data.probki_str_na_liczby(numery_atr)
    print("Dane po konwersji:", data.get_data())


def zad2():
    data = Data()
    data.read_data('spiralka.txt')
    data.probki_str_na_liczby()

    if data.get_data().size == 0:
        print("Błąd: Wczytane dane są puste.")
        return

    kmeans = KMeans(data.get_data(), k=4)
    kmeans.fit()
    
    if kmeans.centroids is not None:
        kmeans.visualize()
        print("Ostateczne punkty centralne:\n", kmeans.get_centroids())

def main():
    zad1()
    zad2()

if __name__ == '__main__':
    np.random.seed(0)
    main()