import numpy as np
import random
import matplotlib.pyplot as plt

def funkcja_przystosowania(x1, x2):
    return np.sin(0.05 * x1) + np.sin(0.05 * x2) + 0.4 * np.sin(0.15 * x1) * np.sin(0.15 * x2)

def pso(N, rinercji, rglob, rlok, iteracje_liczba, xmin, xmax):
    populacja = np.random.uniform(xmin, xmax, (N, 2))
    prędkości = np.zeros((N, 2))
    najlepsze_lokalne = np.copy(populacja)
    oceny_lokalne = np.array([funkcja_przystosowania(ind[0], ind[1]) for ind in najlepsze_lokalne])
    
    ocena_globalna = np.max(oceny_lokalne)
    najlepsza_globalna = najlepsze_lokalne[np.argmax(oceny_lokalne)]

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
        for j in range(N):
            ocena_aktualna = funkcja_przystosowania(populacja[j][0], populacja[j][1])

            if ocena_aktualna > oceny_lokalne[j]:
                oceny_lokalne[j] = ocena_aktualna
                najlepsze_lokalne[j] = populacja[j]

        najlepsza_cząstka = najlepsze_lokalne[np.argmax(oceny_lokalne)]
        ocena_globalna = np.max(oceny_lokalne)

        for j in range(N):
            rndglob = np.random.rand()
            rndlok = np.random.rand()

            for i in range(2):
                prędkości[j, i] = (
                    prędkości[j, i] * rinercji
                    + rglob * rndglob * (najlepsza_cząstka[i] - populacja[j, i])
                    + rlok * rndlok * (najlepsze_lokalne[j, i] - populacja[j, i])
                )
                populacja[j, i] += prędkości[j, i]
            populacja[j] = np.clip(populacja[j], xmin, xmax)

        ax.scatter(populacja[:, 0], populacja[:, 1], [funkcja_przystosowania(x[0], x[1]) for x in populacja], color='red', label=f'Iteracja {t+1}' if t == 0 else "")
        plt.pause(0.1)

        print(f"Iteracja {t+1}: Najlepsze globalne x1 = {najlepsza_cząstka[0]:.4f}, x2 = {najlepsza_cząstka[1]:.4f}, Ocena: {ocena_globalna:.4f}")

    plt.show()

    return najlepsza_cząstka, ocena_globalna

N = 5
rinercji = 0.2
rglob = 0.3
rlok = 0.3
iteracje_liczba = 100
xmin, xmax = 0, 100

najlepsza_cząstka, najlepsza_ocena = pso(N, rinercji, rglob, rlok, iteracje_liczba, xmin, xmax)

print(f"\nNajlepsze rozwiązanie: x1 = {najlepsza_cząstka[0]:.4f}, x2 = {najlepsza_cząstka[1]:.4f}, Ocena: {najlepsza_ocena:.4f}")
