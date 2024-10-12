import numpy as np
import random
import matplotlib.pyplot as plt

def funkcja_przystosowania(x1, x2):
    return np.sin(x1 * 0.05) + np.sin(x2 * 0.05) + 0.4 * np.sin(x1 * 0.15) * np.sin(x2 * 0.15)

def inicjalizuj_populacje(mu, zakres_zmienności):
    return np.random.uniform(zakres_zmienności[0], zakres_zmienności[1], (mu, 2))

def selekcja_turniej(populacja, turniej_rozmiar):
    turniej = random.sample(list(populacja), turniej_rozmiar)
    najlepszy = max(turniej, key=lambda ind: funkcja_przystosowania(ind[0], ind[1]))
    return najlepszy

def krzyzowanie(parent1, parent2):
    point = random.randint(1, 2)
    if point == 1:
        child1 = np.array([parent1[0], parent2[1]])
        child2 = np.array([parent2[0], parent1[1]])
    else:
        child1 = np.array([parent1[0], parent2[1]])
        child2 = np.array([parent2[0], parent1[1]])
    return child1, child2

def mutacja(indywiduum, mutacja_poziom, zakres_zmienności):
    if random.random() < 0.1:  # Prawdopodobieństwo mutacji
        indywiduum[0] += np.random.uniform(-mutacja_poziom, mutacja_poziom)
        indywiduum[1] += np.random.uniform(-mutacja_poziom, mutacja_poziom)
        indywiduum = np.clip(indywiduum, zakres_zmienności[0], zakres_zmienności[1])
    return indywiduum

def algorytm_ewolucyjny(mu, lambd, turniej_rozmiar, mutacja_poziom, iteracje_liczba, zakres_zmienności):
    populacja = inicjalizuj_populacje(mu, zakres_zmienności)
    najlepsze_wyniki = []

    for i in range(iteracje_liczba):
        nowe_indywidua = []
        
        for _ in range(lambd // 2):
            parent1 = selekcja_turniej(populacja, turniej_rozmiar)
            parent2 = selekcja_turniej(populacja, turniej_rozmiar)
            child1, child2 = krzyzowanie(parent1, parent2)
            nowe_indywidua.append(mutacja(child1, mutacja_poziom, zakres_zmienności))
            nowe_indywidua.append(mutacja(child2, mutacja_poziom, zakres_zmienności))
        
        populacja = np.vstack((populacja, nowe_indywidua))
        populacja = sorted(populacja, key=lambda ind: funkcja_przystosowania(ind[0], ind[1]), reverse=True)[:mu]

        najlepsze_wyniki.append(populacja[0])

        print(f"{i + 1} iteracja: Najlepszy osobnik: {populacja[0]}, Wartość przystosowania: {funkcja_przystosowania(populacja[0][0], populacja[0][1])}")

        if i % 5 == 0 or i == iteracje_liczba - 1:  # Co 5 iteracji
            wizualizacja(populacja, najlepsze_wyniki, zakres_zmienności, i)

    najlepszy = max(populacja, key=lambda ind: funkcja_przystosowania(ind[0], ind[1]))
    return najlepszy, najlepsze_wyniki

def wizualizacja(populacja, najlepsze_wyniki, zakres_zmienności, iteracja):
    x1 = np.linspace(zakres_zmienności[0], zakres_zmienności[1], 400)
    x2 = np.linspace(zakres_zmienności[0], zakres_zmienności[1], 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = funkcja_przystosowania(X1, X2)

    plt.figure(figsize=(10, 8))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Wartość funkcji przystosowania')
    
    populacja_np = np.array(populacja)
    plt.scatter(populacja_np[:, 0], populacja_np[:, 1], c='red', marker='o', s=100, label="Osobniki")
    
    najlepszy_osobnik = najlepsze_wyniki[-1]
    plt.scatter(najlepszy_osobnik[0], najlepszy_osobnik[1], c='blue', marker='x', s=200, label="Najlepszy osobnik")
    
    plt.title(f'Iteracja {iteracja + 1}')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

mu = 4
lambd = 10
turniej_rozmiar = 2
mutacja_poziom = 10
iteracje_liczba = 20
zakres_zmienności = (0, 100)

najlepszy_osobnik, najlepsze_wyniki = algorytm_ewolucyjny(mu, lambd, turniej_rozmiar, mutacja_poziom, iteracje_liczba, zakres_zmienności)

x1_wyniki = [ind[0] for ind in najlepsze_wyniki]
x2_wyniki = [ind[1] for ind in najlepsze_wyniki]
y_wyniki = [funkcja_przystosowania(ind[0], ind[1]) for ind in najlepsze_wyniki]

plt.plot(range(1, iteracje_liczba + 1), y_wyniki, marker='o')
plt.title('Najlepsze wyniki w kolejnych iteracjach')
plt.xlabel('Iteracja')
plt.ylabel('Wartość funkcji przystosowania')
plt.xticks(range(1, iteracje_liczba + 1))
plt.grid()
plt.show()

najlepsza_wartosc = funkcja_przystosowania(najlepszy_osobnik[0], najlepszy_osobnik[1])
print(f"\nNajlepszy osobnik: {najlepszy_osobnik}, Wartość funkcji F: {najlepsza_wartosc}")
