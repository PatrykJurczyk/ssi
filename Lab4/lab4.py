import random
import matplotlib.pyplot as plt
import numpy as np

def funkcja_przystosowania(x):
    return np.sin(x / 10) * np.sin(x / 200)

def algorytm_1_plus_1(rozrzut, wsp_przyrostu, l_iteracji, zakres_zmienności=(0, 100)):
    x = random.uniform(zakres_zmienności[0], zakres_zmienności[1])
    y = funkcja_przystosowania(x)
    iteracje = []
    wartosci_x = []
    wartosci_y = []
    
    x_values = np.linspace(zakres_zmienności[0], zakres_zmienności[1], 1000)
    y_values = funkcja_przystosowania(x_values)

    plt.plot(x_values, y_values, label="Funkcja przystosowania")
    
    for i in range(l_iteracji):
        x_pot = x + random.uniform(-rozrzut, rozrzut)

        x_pot = max(min(x_pot, zakres_zmienności[1]), zakres_zmienności[0])
        
        y_pot = funkcja_przystosowania(x_pot)
        
        print(f"Iteracja: {i+1}, x: {x:.4f}, y: {y:.4f}, rozrzut: {rozrzut:.4f}")
        
        if y_pot >= y:
            x, y = x_pot, y_pot
            rozrzut *= wsp_przyrostu
        else:
            rozrzut /= wsp_przyrostu

        iteracje.append(i+1)
        wartosci_x.append(x)
        wartosci_y.append(y)

        plt.scatter(x, y, color='blue')

    plt.scatter(x, y, color='red', label='Najlepszy wynik', zorder=5)
    plt.title("Algorytm 1+1 - Maksymalizacja funkcji")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return x, y

rozrzut = 10
wsp_przyrostu = 1.1
l_iteracji = 100
zakres_zmienności = (0, 100)

najlepszy_x, najlepszy_y = algorytm_1_plus_1(rozrzut, wsp_przyrostu, l_iteracji, zakres_zmienności)

print(f"Najlepszy znaleziony punkt: x = {najlepszy_x:.4f}, y = {najlepszy_y:.4f}")
