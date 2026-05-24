# Wirtualny Sommelier 🍷 | Projekt Analizy i Klasyfikacji Jakości Wina Czerwonego

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Data Analysis](https://img.shields.io/badge/Data%20Analysis-Pandas%20%2B%20NumPy-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![GUI](https://img.shields.io/badge/GUI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Stats](https://img.shields.io/badge/Stats-Statsmodels-005571?style=flat-square)](https://www.statsmodels.org/)
[![Visualization](https://img.shields.io/badge/Visualization-Matplotlib%20%2B%20Seaborn-11557c?style=flat-square)](https://matplotlib.org/)

---

## 1. Cel i Opis Projektu

Projekt **Wirtualny Sommelier** to kompleksowy potok uczenia maszynowego (End-to-End Machine Learning Pipeline) służący do klasyfikacji jakości wina czerwonego na podstawie jego fizykochemicznych parametrów laboratoryjnych. 

Zbiór danych pochodzi z popularnej bazy **Wine Quality (Kaggle)** i zawiera parametry laboratoryjne 1599 próbek portugalskiego wina czerwonego "Vinho Verde".

### Problem badawczy:
Oryginalna zmienna celu `quality` zawiera oceny w skali 3–8. Ze względu na skrajne niezbalansowanie klas (bardzo mało próbek o jakości 3, 4 oraz 8), w projekcie zastosowano metodę **rebinaryzacji** zmiennej celu:
*   **Klasa 1 (Wino Dobre/Premium):** Jakość $\ge 6$ (około 53.4% próbek)
*   **Klasa 0 (Wino Przeciętne/Słabe):** Jakość $< 6$ (około 46.6% próbek)

Dzięki temu przekształceniu problem został sformułowany jako klasyfikacja binarna, co pozwoliło uzyskać wysoką stabilność predykcyjną i doskonałą zdolność generalizacji modelu.

---

## 2. Architektura i Etapy Projektu

Projekt został ustrukturyzowany w sposób modularny, odzwierciedlający najlepsze praktyki inżynierii uczenia maszynowego:

### Etap 1: Zaawansowany Preprocessing i Baseline (`main.py`)
*   **Podział danych (`src/data_loader.py`):** Wczytanie danych, rebinaryzacja oraz podział na zbiór treningowy, walidacyjny i testowy w proporcjach **70% / 15% / 15%** z zachowaniem stratyfikacji (`stratify=y`).
*   **Imputacja braków:** Domyślny potok wykorzystujący zaawansowaną imputację regresyjną MICE (`IterativeImputer`), która automatycznie radzi sobie z ewentualnymi brakami w danych.
*   **Redukcja skośności:** Automatyczne wykrywanie cech o skośności $> 0.75$ i aplikowanie transformacji logarytmicznej (`np.log1p`), co znacząco poprawiło zbieżność modeli liniowych i dystansowych.
*   **Standaryzacja:** Zastosowanie `StandardScaler` dla zapewnienia stabilności numerycznej.
*   **VIF (Variance Inflation Factor):** Analiza współliniowości cech wykazała, że wszystkie parametry mają VIF $\le 10$, dzięki czemu nie było potrzeby usuwania cech ze względu na multikolinearność.
*   **Detekcja Cook's Distance:** Obliczenie odległości Cooka z rygorystycznym progiem $4 / N$ na podstawie regresji logistycznej. Zidentyfikowano i usunięto **67 punktów wpływowych** (outlierów), co podniosło bazową dokładność potoku.

### Etap 2: Walidacja Krzyżowa i Optymalizacja (`etap2_modele.py`)
*   Połączono zbiór treningowy i walidacyjny w celu przeprowadzenia rzetelnej **5-krotnej stratyfikowanej walidacji krzyżowej (Stratified 5-Fold CV)**.
*   Porównano 7 zróżnicowanych algorytmów klasyfikacji (Naiwny Bayes, Drzewo Decyzyjne, Regresja Logistyczna, K-NN, SVM z jądrem RBF, Gradient Boosting, Las Losowy).
*   Championem walidacji okazał się **Las Losowy (Random Forest)** z wynikiem **80.65%**.
*   Wdrożono automatyczne **strojenie hiperparametrów za pomocą `GridSearchCV`** dla zwycięskiego modelu.
*   **Optymalne parametry:** `{'n_estimators': 100, 'min_samples_split': 5, 'max_depth': None}`.
*   Dzięki strojeniu dokładność CV wzrosła do **`81.31%`** (poprawa o **+0.66%**).

### Etap 3: Ewaluacja Końcowa i Diagnostyka (`etap3_ewaluacja.py`)
*   Wytrenowano ostateczny Pipeline na pełnym zbiorze treningowym (po usunięciu 77 outlierów na połączonym zbiorze trening+val) przy użyciu optymalnych hiperparametrów.
*   Przetestowano model na całkowicie odizolowanym zbiorze testowym (240 próbek), uzyskując ostateczną celność **`76.25%`**.
*   Wygenerowano i zapisano trzy kluczowe wykresy diagnostyczne:
    1.  `confusion_matrix.png` – Macierz pomyłek wskazująca na świetny balans predykcji.
    2.  `roc_curve.png` – Krzywa ROC z wysokim współczynnikiem **AUC = 0.84** potwierdzającym świetną jakość dyskryminacyjną klasyfikatora.
    3.  `learning_curve.png` – Krzywe uczenia pokazujące dynamikę dokładności treningowej i walidacyjnej, wykluczające overfitting.

### Etap 4: Eksport i Analiza Fizykochemiczna (`etap4_eksport.py`)
*   Obliczono ważność cech (*Feature Importance*) na pełnym zbiorze danych, uzyskując kluczowe wnioski oenologiczne.
*   Wyłoniono **TOP 5 cech** determinujących jakość czerwonego wina.
*   Wytrenowano lekki model (Lightweight Pipeline) oparty wyłącznie o te 5 zmiennych laboratoryjnych i wyeksportowano go do pliku `wine_model_light.pkl`.

### Etap 5: Produkcyjna Aplikacja Webowa (`app.py`)
*   Zaimplementowano elegancki interfejs webowy przy użyciu frameworka **Streamlit**.
*   Zastosowano motyw **Premium Dark Sommelier** z głęboką czerwienią burgundu (`#722F37`) oraz złotymi akcentami (`#D4AF37`).
*   Użytkownik wprowadza wartości 5 kluczowych cech fizykochemicznych za pomocą precyzyjnych suwaków ze zweryfikowanymi fizjologicznie zakresami, a model w czasie rzeczywistym szacuje prawdopodobieństwo klasy **Premium** wraz z efektami wizualnymi (balony przy sukcesie).

---

## 3. Struktura Repozytorium

```text
VirtualSommelierML/
│
├── data/
│   └── winequality-red.csv       # Surowy zbiór danych wina czerwonego (z Kaggle)
│
├── src/
│   ├── __init__.py
│   └── data_loader.py            # Moduł ładowania danych, rebinaryzacji i stratyfikowanego podziału
│
├── main.py                       # ETAP 1: Czyszczenie danych, badanie baseline i outlierów (Cook's D)
├── etap2_modele.py               # ETAP 2: Stratified 5-Fold CV dla 7 modeli + GridSearchCV dla Random Forest
├── etap3_ewaluacja.py            # ETAP 3: Ewaluacja końcowa zbioru testowego, generowanie grafik diagnostycznych
├── etap4_eksport.py              # ETAP 4: Wyłonienie TOP 5 cech i eksport lekkiego modelu do pliku .pkl
├── app.py                        # ETAP 5: Aplikacja webowa GUI (Streamlit) - Wirtualny Sommelier
│
├── wine_model_light.pkl          # Serializowany lekki model wdrożeniowy (generowany w Etapie 4)
├── confusion_matrix.png          # Wykres macierzy pomyłek na zbiorze testowym
├── roc_curve.png                 # Wykres krzywej ROC (AUC) na zbiorze testowym
├── learning_curve.png            # Wykres krzywych uczenia (diagnoza bias/variance)
├── README.md                     # Dokumentacja projektu
└── .gitignore
```

---

## 4. Wyniki Modelu i Analiza

### Ewaluacja końcowa (Zbiór Testowy):
*   **Ostateczna dokładność (Accuracy): 76.25%**
*   **Klasa 0 (Przeciętne/Słabe):** F1-Score: **0.75** (Precyzja: 74%, Czułość: 76%)
*   **Klasa 1 (Dobre/Premium):** F1-Score: **0.77** (Precyzja: 78%, Czułość: 77%)

Model charakteryzuje się niemal symetryczną skutecznością predykcji zarówno dla win przeciętnych, jak i dobrych (bardzo zbliżone wskaźniki F1-Score).

### TOP 5 cech fizykochemicznych wina (Feature Importance):
1.  **`alcohol` (20.1%)** – Zawartość alkoholu. Najbardziej decydujący czynnik wpływający na pełnię i strukturę wina. Wyższa zawartość alkoholu silnie koreluje z wyższą oceną jakości.
2.  **`sulphates` (13.0%)** – Siarczany. Działają jako przeciwutleniacz i konserwant. Odpowiednia ich obecność pozwala zachować świeżość i chroni wino przed psuciem.
3.  **`volatile acidity` (11.0%)** – Kwasowość lotna. Reprezentuje ilość kwasu octowego. Zbyt wysoki poziom daje nieprzyjemny, octowy zapach i smak, stąd niska kwasowość lotna silnie sprzyja klasie Premium.
4.  **`total sulfur dioxide` (10.0%)** – Całkowity dwutlenek siarki. Kluczowy dla stabilności mikrobiologicznej wina.
5.  **`density` (8.5%)** – Gęstość wina. Silnie powiązana z zawartością cukru resztkowego i alkoholu w badanej próbce.

---

## 5. Instrukcja Instalacji i Uruchomienia

### Krok 1: Przygotowanie środowiska wirtualnego
Upewnij się, że posiadasz zainstalowany interpreter Python w wersji **3.9+**. Utwórz i aktywuj środowisko wirtualne:

```bash
# Utworzenie środowiska wirtualnego
python3 -m venv .venv

# Aktywacja środowiska wirtualnego (macOS / Linux)
source .venv/bin/activate

# Alternatywna aktywacja w systemie Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### Krok 2: Instalacja wymaganych bibliotek
Zainstaluj zależności za pomocą menedżera pakietów `pip`:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn joblib streamlit
```

### Krok 3: Wygenerowanie i eksport modelu ML
Przed pierwszym uruchomieniem aplikacji webowej należy wytrenować i wyeksportować lekki model:

```bash
python3 etap4_eksport.py
```

Skrypt wyliczy ważność cech i wyeksportuje plik `wine_model_light.pkl` w głównym katalogu projektu.

### Krok 4: Uruchomienie aplikacji webowej Streamlit
Aby otworzyć interfejs **Wirtualnego Sommeliera** w przeglądarce internetowej, wpisz w terminalu:

```bash
streamlit run app.py
```

Aplikacja automatycznie otworzy nową kartę pod lokalnym adresem sieciowym (domyślnie `http://localhost:8501`).

---
*Projekt końcowy przygotowany w ramach przedmiotu Uczenie Maszynowe II.*
