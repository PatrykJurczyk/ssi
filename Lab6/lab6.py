import numpy as np
import matplotlib.pyplot as plt


class GreedyPointMatching:
    def __init__(self, main_bitmap):
        self.main_bitmap = np.array(main_bitmap)
        self.test_bitmaps = {}
        self.similarity_measures = {}

    def add_test_bitmap(self, new_bitmap, label):
        self.test_bitmaps[label] = np.array(new_bitmap)
        self.similarity_measures[label] = -np.inf

    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    def dissimilarity_measures(self, bit_a, bit_b):
        total_distance = 0
        for (pay, pax) in np.argwhere(bit_a == 1):
            min_distance = np.inf
            for (pby, pbx) in np.argwhere(bit_b == 1):
                current_distance = self.euclidean_distance([pax, pay], [pbx, pby])
                min_distance = min(min_distance, current_distance)
            total_distance += min_distance
        return total_distance

    def two_sided_similarity_measures(self, bit_a, bit_b):
        return -(self.dissimilarity_measures(bit_a, bit_b) + self.dissimilarity_measures(bit_b, bit_a))

    def calculate_similarities(self):
        for label, test_bitmap in self.test_bitmaps.items():
            self.similarity_measures[label] = self.two_sided_similarity_measures(self.main_bitmap, test_bitmap)

    def visualize(self):
        num_test_bitmaps = len(self.test_bitmaps)
        fig, axes = plt.subplots(2, num_test_bitmaps + 1, figsize=(12, 6))

        axes[0, 0].imshow(self.main_bitmap, cmap='Greys')
        axes[0, 0].set_title('Bitmap Główna')
        axes[0, 0].axis('off')

        for idx, (label, bitmap) in enumerate(self.test_bitmaps.items()):
            axes[0, idx + 1].imshow(bitmap, cmap='Greys')
            axes[0, idx + 1].set_title(f'Test: {label}')
            axes[0, idx + 1].axis('off')

        for idx, label in enumerate(self.similarity_measures.keys()):
            axes[1, idx].text(0.5, 0.5, f'{label}\n{self.similarity_measures[label]:.2f}',
                              horizontalalignment='center', verticalalignment='center', fontsize=14)
            axes[1, idx].axis('off')

        best_match_label = max(self.similarity_measures, key=self.similarity_measures.get)
        best_match_value = self.similarity_measures[best_match_label]

        axes[1, num_test_bitmaps].text(0.5, 0.5, f'Najlepsze dopasowanie:\n{best_match_label}\n{best_match_value:.2f}',
                                         horizontalalignment='center', verticalalignment='center', fontsize=14)
        axes[1, num_test_bitmaps].axis('off')

        plt.tight_layout()
        plt.show()

    def __call__(self, visualize=True):
        self.calculate_similarities()
        if visualize:
            self.visualize()
        else:
            print('Brak obrazów testowych do porównania.')


def zad1():
    test1 = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
    test2 = [[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
    test3 = [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    wzorzec1 = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
    wzorzec2 = [[0, 1, 1, 1], [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]]
    wzorzec3 = [[1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 0]]

    gpm1 = GreedyPointMatching(test1)
    gpm1.add_test_bitmap(wzorzec1, 'Wzorzec 1')
    gpm1.add_test_bitmap(wzorzec2, 'Wzorzec 2')
    gpm1.add_test_bitmap(wzorzec3, 'Wzorzec 3')
    gpm1()

    gpm2 = GreedyPointMatching(test2)
    gpm2.add_test_bitmap(wzorzec1, 'Wzorzec 1')
    gpm2.add_test_bitmap(wzorzec2, 'Wzorzec 2')
    gpm2.add_test_bitmap(wzorzec3, 'Wzorzec 3')
    gpm2()

    gpm3 = GreedyPointMatching(test3)
    gpm3.add_test_bitmap(wzorzec1, 'Wzorzec 1')
    gpm3.add_test_bitmap(wzorzec2, 'Wzorzec 2')
    gpm3.add_test_bitmap(wzorzec3, 'Wzorzec 3')
    gpm3()


def main():
    zad1()


if __name__ == '__main__':
    main()
