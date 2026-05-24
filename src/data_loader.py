import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_wine_data(file_path):
    """
    Wczytuje zbiór danych o winie, przeprowadza czyszczenie braków (zamiana '?' na NaN),
    wykonuje rebinaryzację zmiennej docelowej 'quality' (1 dla >= 6, 0 dla < 6)
    oraz dokonuje stratyfikowanego podziału na zbiór treningowy (70%), walidacyjny (15%) i testowy (15%).
    """
    df = pd.read_csv(file_path)

    # Zamiana znaków zapytania na NaN i konwersja na typy numeryczne
    df.replace('?', np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rebinaryzacja zmiennej docelowej quality: 1 (Dobre/Premium >= 6), 0 (Przeciętne/Słabe < 6)
    df['quality'] = (df['quality'] >= 6).astype(int)

    X = df.drop('quality', axis=1)
    y = df['quality']

    # Pierwszy podział: 70% trening / 30% tymczasowy
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Drugi podział: 50% z 30% na walidacyjny (15%) i 50% na testowy (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(base_dir, '..', 'data', 'winequality-red.csv'))

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_wine_data(data_path)
        print("--- SUKCES: Dane wczytane i podzielone poprawnie ---")
        print(f"Rozmiary zbiorów: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")
        print(f"Rozkład klas w zbiorze treningowym:\n{y_train.value_counts(normalize=True).round(3)}")
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku pod ścieżką: {data_path}")
