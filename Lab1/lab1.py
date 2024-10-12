import pandas as pd
import re

def wczytaj_baze_probek_z_tekstem(nazwa_pliku_z_wartosciami, nazwa_pliku_z_opisem_atr):
    try:
        atr_data = pd.read_csv(nazwa_pliku_z_opisem_atr, sep=r'\s+', header=None, names=['nazwa_atr', 'typ_atr'])
        nazwy_atr = atr_data['nazwa_atr'].tolist()
        czy_atr_symb = atr_data['typ_atr'].apply(lambda x: x == 's').tolist()

        df_probki = pd.read_csv(nazwa_pliku_z_wartosciami, sep=r'\s+', header=None, names=nazwy_atr)
        
        symboliczna_kolumna = nazwy_atr[czy_atr_symb.index(True)]
        match = re.search(r'class\((.+)\)', symboliczna_kolumna)

        if match:
            mapowanie_class = dict(item.split('=') for item in match.group(1).split(','))
            df_probki[symboliczna_kolumna] = df_probki[symboliczna_kolumna].apply(lambda x: mapowanie_class.get(str(x), x))

        return df_probki, czy_atr_symb, nazwy_atr

    except FileNotFoundError as e:
        print(f"Błąd: {e}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")


df, czy_atr_symb, nazwy_atr = wczytaj_baze_probek_z_tekstem('iris.txt', 'iris-type.txt')
print("Tabela próbek:")
print(df)