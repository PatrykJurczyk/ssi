import numpy as np
import matplotlib.pyplot as plt

def funkcja_przystosowania(x1, x2):
    return np.sin(0.05 * x1) + np.sin(0.05 * x2) + 0.4 * np.sin(0.15 * x1) * np.sin(0.15 * x2)

def odleglosc(firefly_a, firefly_b):
    return np.sqrt(np.sum((firefly_a - firefly_b) ** 2))

def firefly_algorithm(N, beta0, gamma0, mu0, iteracje_liczba, xmin, xmax):
    populacja = np.random.uniform(xmin, xmax, (N, 2))
    oceny = np.array([funkcja_przystosowania(ind[0], ind[1]) for ind in populacja])
    
    r_max = np.sqrt(2 * (xmax - xmin) ** 2)
    gamma = gamma0 / r_max

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    x1_vals = np.linspace(xmin, xmax, 100)
    x2_vals = np.linspace(xmin, xmax, 100)
    x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
    z_vals = funkcja_przystosowania(x1_grid, x2_grid)

    ax.plot_surface(x1_grid, x2_grid, z_vals, cmap='viridis', alpha=0.6)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('F(x1, x2)')

    for t in range(iteracje_liczba):
        for a in range(N):
            for b in range(N):
                if oceny[b] > oceny[a]:
                    r_ab = odleglosc(populacja[a], populacja[b])
                    beta = beta0 * np.exp(-gamma * r_ab ** 2)
                    populacja[a] += beta * (populacja[b] - populacja[a])
                    mutacja = mu0 * (xmax - xmin) * (np.random.rand(2) - 0.5)
                    populacja[a] += mutacja

                    populacja[a] = np.clip(populacja[a], xmin, xmax)

            oceny[a] = funkcja_przystosowania(populacja[a][0], populacja[a][1])

        ax.scatter(populacja[:, 0], populacja[:, 1], oceny, color='red', label=f'Iteracja {t+1}' if t == 0 else "")
        plt.pause(0.1)

        najlepszy_indeks = np.argmax(oceny)
        print(f"Iteracja {t+1}: Najlepsze x1 = {populacja[najlepszy_indeks][0]:.4f}, x2 = {populacja[najlepszy_indeks][1]:.4f}, Ocena: {oceny[najlepszy_indeks]:.4f}")
    plt.show()

    najlepszy_indeks = np.argmax(oceny)
    return populacja[najlepszy_indeks], oceny[najlepszy_indeks]

N = 5
beta0 = 0.3
gamma0 = 0.1
mu0 = 0.05
iteracje_liczba = 100
xmin, xmax = 0, 100

najlepszy_swietlik, najlepsza_ocena = firefly_algorithm(N, beta0, gamma0, mu0, iteracje_liczba, xmin, xmax)

print(f"\nNajlepsze rozwiązanie: x1 = {najlepszy_swietlik[0]:.4f}, x2 = {najlepszy_swietlik[1]:.4f}, Ocena: {najlepsza_ocena:.4f}")
