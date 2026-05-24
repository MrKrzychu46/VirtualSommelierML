# Przewodnik po Etapach Projektu Uczenia Maszynowego

Poniższy opis wyjaśnia krok po kroku każdy etap wdrożony w projekcie **Wirtualny Sommelier**. Został przygotowany z perspektywy programistycznej: wyjaśnia **Co** robimy, **Po co** to robimy (cel biznesowy/techniczny) oraz **Jak** to realizujemy w kodzie wraz z przykładami.

---

## Spis treści
1. Podział danych (Train, Validation, Test)
2. Zarządzanie brakami (Imputacja)
3. Transformacja Logarytmiczna (log1p)
4. Standaryzacja Cech (StandardScaler)
5. Redukcja Współliniowości (VIF)
6. Obserwacje Wpływowe (Odległość Cooka)
7. Potoki (Pipelines) i Walidacja Krzyżowa (Cross-Validation)
8. Strojenie Parametrów (Grid Search)
9. Metryki Oceny (Accuracy, Precision, Recall, F1)
10. Wizualizacje (Confusion Matrix, ROC/AUC, Learning Curves)
11. Istotność Cech i Model Lekki (Eksport)

---

## 1. Podział danych (Train, Validation, Test)

*   **Co:** Dzielimy cały posiadany zbiór danych na trzy osobne części: treningową (70%), walidacyjną (15%) i testową (15%). Podział jest *stratyfikowany* (zachowuje proporcje klas).
*   **Po co:** 
    *   **Zbiór treningowy** służy do nauki modeli.
    *   **Zbiór walidacyjny** służy do porównywania modeli i strojenia ich parametrów (pomaga wybrać najlepszą konfigurację).
    *   **Zbiór testowy** służy do ostatecznej oceny. Musi być całkowicie odcięty od procesu uczenia, aby zapobiec przeuczeniu (*overfitting*).
*   **Jak:**
    ```python
    from sklearn.model_selection import train_test_split

    # Podział 70% trening / 30% tymczasowy
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Drugi podział: 50% z 30% na walidacyjny (15%) i testowy (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    ```

---

## 2. Zarządzanie brakami (Imputacja)

*   **Co:** Uzupełnianie brakujących komórek w tabeli (wartości `NaN` / `None`).
*   **Po co:** Większość algorytmów uczenia maszynowego nie potrafi obsłużyć brakujących wartości i rzuca wyjątkami.
*   **Jak:** Używamy `SimpleImputer` (mediana) lub `IterativeImputer` (regresja MICE):
    ```python
    from sklearn.impute import SimpleImputer, IterativeImputer

    # Proste uzupełnianie medianą
    imputer_med = SimpleImputer(strategy='median')
    X_train_imp_med = imputer_med.fit_transform(X_train)

    # Zaawansowana imputacja regresyjna (MICE)
    imputer_reg = IterativeImputer(random_state=42, max_iter=10)
    X_train_imp_reg = imputer_reg.fit_transform(X_train)
    ```

---

## 3. Transformacja Logarytmiczna (log1p)

*   **Co:** Zastosowanie funkcji matematycznej $f(x) = \ln(x + 1)$ na cechach o wysokiej skośności.
*   **Po co:** Logarytmowanie "ściąga" skrajne wartości bliżej centrum, sprowadzając rozkład zmiennych do zbliżonego do normalnego.
*   **Jak:**
    ```python
    # Wyznaczenie cech prawoskośnych
    skew_vals = X_train.skew()
    skewed_features = skew_vals[abs(skew_vals) > 0.75].index.tolist()

    # Zastosowanie log1p zabezpieczonego przed wartościami ujemnymi
    for col in skewed_features:
        X_train[col] = np.log1p(np.clip(X_train[col], 0, None))
    ```

---

## 4. Standaryzacja Cech (StandardScaler)

*   **Co:** Przeskalowanie wszystkich cech tak, aby ich średnia wynosiła `0`, a odchylenie standardowe `1`.
*   **Po co:** Zapobiega dominacji modelu przez cechy posiadające naturalnie większe zakresy liczbowe (np. wolny dwutlenek siarki w skali 1-70 vs kwasowość w skali 0.1-1.5).
*   **Jak:**
    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Używamy parametrów dopasowanych na treningowym!
    ```

---

## 5. Redukcja Współliniowości (VIF - Variance Inflation Factor)

*   **Co:** Wykrywanie i usuwanie cech nadmiarowych, które są silnie skorelowane z innymi cechami.
*   **Po co:** Wielokollinearność cech nadmiernie zwiększa wariancję modelu i destabilizuje wagi modeli liniowych.
*   **Jak:**
    ```python
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    def compute_vif(df_vif):
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(len(df_vif.columns))]
        return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    # Iteracyjne usuwanie cech o VIF > 10
    vif_df = compute_vif(X_train)
    if vif_df.iloc[0]["VIF"] > 10:
        X_train = X_train.drop(columns=[vif_df.iloc[0]["feature"]])
    ```

---

## 6. Obserwacje Wpływowe (Odległość Cooka)

*   **Co:** Identyfikacja i eliminacja anomalii ze zbioru treningowego, które zaburzają naukę modelu.
*   **Po co:** Chroni model przed dopasowywaniem się do błędnych lub bardzo nietypowych obserwacji (outlierów).
*   **Jak:**
    ```python
    import statsmodels.api as sm

    X_train_const = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_train_const).fit(disp=0)
    cooks_d = logit_model.get_influence().cooks_distance[0]

    # Usunięcie punktów powyżej progu 4 / N
    cook_threshold = 4.0 / len(X_train)
    safe_indices = np.where(cooks_d <= cook_threshold)[0]
    X_train_clean = X_train.iloc[safe_indices]
    y_train_clean = y_train.iloc[safe_indices]
    ```

---

## 7. Potoki (Pipelines) i Walidacja Krzyżowa (Cross-Validation)

*   **Co:** Łączenie transformacji danych i klasyfikatora w jeden potok oraz podział na $k$ foldów w celu stabilnej oceny.
*   **Po co:** Pipeline gwarantuje, że nie dojdzie do wycieku informacji ze zbioru walidacyjnego do treningowego (brak wycieku danych - *data leakage*).
*   **Jak:**
    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # Budowa potoku
    pipeline = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 5-krotna walidacja krzyżowa
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='accuracy')
    print(f"Średnia dokładność (CV): {scores.mean():.4f}")
    ```

---

## 8. Strojenie Parametrów (Grid Search)

*   **Co:** Przeszukiwanie siatki hiperparametrów w celu optymalizacji konfiguracji modelu.
*   **Po co:** Każdy algorytm posiada parametry, które musimy ustawić ręcznie. Grid Search znajduje najlepszą ich kombinację.
*   **Jak:**
    ```python
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 15]
    }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=kf, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Najlepsze parametry: {grid_search.best_params_}")
    ```

---

## 9. Metryki Oceny Klasyfikacji

*   **Co:** Ostateczny test na izolowanym zbiorze testowym w celu weryfikacji metryk jakości.
*   **Po co:** Raport klasyfikacji dostarcza szczegółowych informacji o zachowaniu modelu dla poszczególnych klas.
*   **Jak:**
    ```python
    from sklearn.metrics import classification_report

    y_pred = final_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Słabe", "Dobre"]))
    ```

---

## 10. Wizualizacje

Wykorzystujemy trzy kluczowe wykresy do graficznej analizy modelu:

### Macierz pomyłek (Confusion Matrix)
*   Pokazuje liczbowo rzeczywiste vs przewidziane klasy.
    ```python
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Słabe", "Dobre"])
    disp.plot(cmap='Blues')
    plt.show()
    ```

### Krzywa ROC i AUC
*   Pokazuje czułość modelu w funkcji odsetka fałszywych alarmów przy różnych progach odcięcia.
    ```python
    from sklearn.metrics import roc_curve, auc

    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.legend()
    plt.show()
    ```

### Krzywe uczenia (Learning Curves)
*   Pozwala zdiagnozować niedouczenie (underfitting) lub przeuczenie (overfitting) w zależności od rozmiaru danych.
    ```python
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=final_pipeline, X=X_train, y=y_train, cv=5
    )
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Trening')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Walidacja')
    plt.legend()
    plt.show()
    ```

---

## 11. Istotność Cech i Model Lekki (Eksport)

*   **Co:** Wyznaczenie najważniejszych cech, wytrenowanie na nich uproszczonego modelu oraz jego eksport.
*   **Po co:** Ograniczenie liczby cech (np. do TOP 5) upraszcza wdrożenie modelu na produkcji oraz ułatwia użytkowanie aplikacji (użytkownik podaje tylko 5 wartości zamiast 11).
*   **Jak:**
    ```python
    import joblib

    # 1. Odczyt ważności cech
    importances = final_pipeline.named_steps['classifier'].feature_importances_
    
    # 2. Trening lekkiego modelu na TOP 5 cechach
    top_features = ['alcohol', 'sulphates', 'volatile acidity', 'total sulfur dioxide', 'density']
    light_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    light_pipeline.fit(X_full[top_features], y_full)
    
    # 3. Zapis modelu do pliku .pkl
    joblib.dump((light_pipeline, top_features), 'wine_model_light.pkl')
    ```

---

## 12. Podsumowanie i Wnioski

*   **Wpływ preprocessingu:** Etapy inżynierii cech z Etapu 1 (w szczególności standaryzacja cech oraz eliminacja obserwacji wpływowych metodą Cooka) przyniosły kluczową poprawę wydajności. Dokładność (accuracy) baseline'owego modelu K-NN wzrosła dzięki tym krokom z początkowego poziomu `0.6417` do ostatecznego `0.7583`.
*   **Wybór modelu:** Algorytm Lasu Losowego (Random Forest) okazał się bezkonkurencyjnym zwycięzcą walidacji krzyżowej, deklasując modele liniowe (regresja logistyczna, SVM) oraz prostsze drzewiaste (pojedyncze drzewo decyzyjne). Strojenie hiperparametrów za pomocą `GridSearchCV` przyniosło dalszy wzrost dokładności na zbiorze walidacyjnym do poziomu `0.8146`.
*   **Generalizacja (Zbiór testowy):** Ostateczny model osiągnął stabilną dokładność `0.7625` na całkowicie odciętym zbiorze testowym, co potwierdza wysoką zdolność do generalizacji na nieznanych wcześniej próbkach.
*   **Znaczenie biznesowe i wdrożenie:**
    1.  **Interpretacja chemiczna:** Model potwierdził, że najważniejszymi fizykochemicznymi predyktorami jakości wina są: zawartość alkoholu (istotność ~20%), zawartość siarczanów (~13%) oraz kwasowość lotna (~11%).
    2.  **Lekki model (Lightweight RF):** Ograniczenie modelu do TOP 5 cech z prostą imputacją medianą pozwala na łatwe zintegrowanie modelu z aplikacją okienkową lub API. Dzięki temu użytkownik musi wprowadzić tylko podstawowe wskaźniki, a model zachowuje wysoką precyzję działania bez konieczności pełnego zestawu 11 parametrów.
