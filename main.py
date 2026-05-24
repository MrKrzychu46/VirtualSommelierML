import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.data_loader import load_and_split_wine_data

LICZBA_SASIADOW = 9

def test_knn(X_train, X_val, y_train, y_val, label=""):
    """Szybki test baseline K-NN w celu oceny wpływu inżynierii cech."""
    knn = KNeighborsClassifier(n_neighbors=LICZBA_SASIADOW)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_val, knn.predict(X_val))
    print(f" [k-NN Accuracy: {label}] -> {acc:.4f}")
    return acc

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(base_dir, 'data', 'winequality-red.csv'))

    print("Wczytywanie i podział danych...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_wine_data(data_path)

    # Pracujemy na cechach numerycznych (w winie wszystkie cechy są numeryczne)
    X_train_num = X_train.select_dtypes(include=[np.number])
    X_val_num = X_val.select_dtypes(include=[np.number])
    X_test_num = X_test.select_dtypes(include=[np.number])

    print("\n--- Etap 1a: Zarządzanie brakami danych ---")
    braki = X_train_num.isnull().sum()
    print(f"Liczba braków w kolumnach:\n{braki[braki > 0] if braki.sum() > 0 else 'Brak brakujących wartości.'}")

    # Strategia 1: Usuwanie braków
    X_train_drop = X_train_num.dropna()
    y_train_drop = y_train.loc[X_train_drop.index]
    X_val_drop = X_val_num.dropna()
    y_val_drop = y_val.loc[X_val_drop.index]

    if len(X_train_drop) > 0 and len(X_val_drop) > 0:
        acc_drop = test_knn(X_train_drop, X_val_drop, y_train_drop, y_val_drop, "Po usunięciu")
    else:
        acc_drop = 0.0

    # Strategia 2: Imputacja medianą
    imputer_med = SimpleImputer(strategy='median')
    X_train_imp_med = pd.DataFrame(imputer_med.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    X_val_imp_med = pd.DataFrame(imputer_med.transform(X_val_num), columns=X_val_num.columns, index=X_val_num.index)
    acc_imp_med = test_knn(X_train_imp_med, X_val_imp_med, y_train, y_val, "Po medianie")

    # Strategia 3: Imputacja regresyjna (IterativeImputer MICE)
    imputer_reg = IterativeImputer(random_state=42, max_iter=10)
    X_train_imp_reg = pd.DataFrame(imputer_reg.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index)
    X_val_imp_reg = pd.DataFrame(imputer_reg.transform(X_val_num), columns=X_val_num.columns, index=X_val_num.index)
    acc_imp_reg = test_knn(X_train_imp_reg, X_val_imp_reg, y_train, y_val, "Po regresji")

    # Wybór najlepszej strategii imputacji
    wyniki_imputacji = {
        "Usunięcie braków": (acc_drop, "drop"),
        "Imputacja medianą": (acc_imp_med, "median"),
        "Imputacja regresyjna": (acc_imp_reg, "regression")
    }

    najlepsza_metoda = max(wyniki_imputacji, key=lambda k: wyniki_imputacji[k][0])
    val_acc = wyniki_imputacji[najlepsza_metoda][0]
    typ_imputacji = wyniki_imputacji[najlepsza_metoda][1]

    if typ_imputacji == "drop":
        X_train_final, y_train_final = X_train_drop, y_train_drop
        X_val_final, y_val_final = X_val_drop, y_val_drop
        X_test_final = X_test_num.dropna()
        y_test_final = y_test.loc[X_test_final.index]
    elif typ_imputacji == "median":
        X_train_final, y_train_final = X_train_imp_med, y_train
        X_val_final, y_val_final = X_val_imp_med, y_val
        X_test_final = pd.DataFrame(imputer_med.transform(X_test_num), columns=X_test_num.columns, index=X_test_num.index)
        y_test_final = y_test
    else:
        X_train_final, y_train_final = X_train_imp_reg, y_train
        X_val_final, y_val_final = X_val_imp_reg, y_val
        X_test_final = pd.DataFrame(imputer_reg.transform(X_test_num), columns=X_test_num.columns, index=X_test_num.index)
        y_test_final = y_test

    print(f"✅ [ZATWIERDZONO IMPUTACJĘ: {najlepsza_metoda} | Baseline Accuracy: {val_acc:.4f}]")

    print("\n--- Etap 1b: Analiza skośności i logarytmowanie ---")
    skew_threshold = 0.75
    skew_vals = X_train_final.skew()
    skewed_features = skew_vals[abs(skew_vals) > skew_threshold].index.tolist()
    print(f"Cechy wytypowane do transformacji (skośność > {skew_threshold}): {skewed_features}")

    X_train_trans = X_train_final.copy()
    X_val_trans = X_val_final.copy()
    X_test_trans = X_test_final.copy()

    for col in skewed_features:
        X_train_trans[col] = np.log1p(np.clip(X_train_trans[col], 0, None))
        X_val_trans[col] = np.log1p(np.clip(X_val_trans[col], 0, None))
        X_test_trans[col] = np.log1p(np.clip(X_test_trans[col], 0, None))

    acc_trans = test_knn(X_train_trans, X_val_trans, y_train_final, y_val_final, "Po transformacji log1p")

    if acc_trans > val_acc:
        X_train_final, X_val_final, X_test_final = X_train_trans, X_val_trans, X_test_trans
        print(f"✅ [ZATWIERDZONO: Transformacja logarytmiczna (Poprawa z {val_acc:.4f} na {acc_trans:.4f})]")
        val_acc = acc_trans
    else:
        print("❌ [ODRZUCONO: Transformacja logarytmiczna nie poprawiła wyniku. Zachowano oryginalne cechy.]")

    print("\n--- Etap 1c: Standaryzacja cech (StandardScaler) ---")
    scaler = StandardScaler()
    X_train_std = pd.DataFrame(scaler.fit_transform(X_train_final), columns=X_train_final.columns, index=X_train_final.index)
    X_val_std = pd.DataFrame(scaler.transform(X_val_final), columns=X_val_final.columns, index=X_val_final.index)
    X_test_std = pd.DataFrame(scaler.transform(X_test_final), columns=X_test_final.columns, index=X_test_final.index)

    acc_std = test_knn(X_train_std, X_val_std, y_train_final, y_val_final, "Po standaryzacji")

    if acc_std > val_acc:
        X_train_final, X_val_final, X_test_final = X_train_std, X_val_std, X_test_std
        print(f"✅ [ZATWIERDZONO: Standaryzacja (Poprawa z {val_acc:.4f} na {acc_std:.4f})]")
        val_acc = acc_std
    else:
        print("❌ [ODRZUCONO: Standaryzacja nie poprawiła wyniku. Zachowano nieprzeskalowane cechy.]")

    print("\n--- Etap 1d: Redukcja współliniowości (VIF) ---")
    def compute_vif(df_vif):
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(len(df_vif.columns))]
        return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    X_train_vif = X_train_final.copy()
    X_val_vif = X_val_final.copy()
    X_test_vif = X_test_final.copy()
    removed_vif = []
    iteration = 1
    vif_threshold = 10.0

    print("Iteracyjna weryfikacja współczynników VIF...")
    while True:
        try:
            vif_df = compute_vif(X_train_vif)
            max_vif = vif_df.iloc[0]["VIF"]
            col_to_remove = vif_df.iloc[0]["feature"]

            if max_vif > vif_threshold:
                print(f" Iteracja {iteration}: Usuwam cechę '{col_to_remove}' (VIF = {max_vif:.2f})")
                removed_vif.append(col_to_remove)
                X_train_vif = X_train_vif.drop(columns=[col_to_remove])
                X_val_vif = X_val_vif.drop(columns=[col_to_remove])
                X_test_vif = X_test_vif.drop(columns=[col_to_remove])
                iteration += 1
            else:
                print(f" Koniec: Wszystkie cechy mają VIF <= {vif_threshold}.")
                break
        except Exception as e:
            print(f" Błąd podczas obliczania VIF: {e}. Przerywam proces.")
            break

    if len(removed_vif) > 0:
        acc_vif = test_knn(X_train_vif, X_val_vif, y_train_final, y_val_final, "Po redukcji VIF")
        if acc_vif >= val_acc:
            X_train_final, X_val_final, X_test_final = X_train_vif, X_val_vif, X_test_vif
            print(f"✅ [ZATWIERDZONO: Redukcja VIF (Wynik: {acc_vif:.4f}, Usunięto: {removed_vif})]")
            val_acc = acc_vif
        else:
            print(f"❌ [ODRZUCONO: Redukcja VIF pogorszyła model baseline (Spadek do {acc_vif:.4f}). Przywracam cechy.]")
    else:
        print("--- Brak współliniowych cech do usunięcia.")

    print("\n--- Etap 1e: Detekcja obserwacji wpływowych (Odległość Cooka) ---")
    try:
        X_train_const = sm.add_constant(X_train_final)
        # Ponieważ y jest zmienną binarną, używamy modelu logitowego do Cook'a
        logit_model = sm.Logit(y_train_final.values, X_train_const).fit(disp=0)
        influence = logit_model.get_influence()
        cooks_d = influence.cooks_distance[0]

        # W notebooku próg to 4 / len(X_train_final)
        cook_threshold = 4.0 / len(X_train_final)
        safe_indices = np.where(cooks_d <= cook_threshold)[0]
        n_influential = len(cooks_d) - len(safe_indices)

        print(f"Liczba punktów wpływowych (Cook's Distance > {cook_threshold:.5f}): {n_influential} ({n_influential/len(X_train_final):.1%})")

        if n_influential > 0:
            X_train_clean = X_train_final.iloc[safe_indices]
            y_train_clean = y_train_final.iloc[safe_indices]

            acc_clean = test_knn(X_train_clean, X_val_final, y_train_clean, y_val_final, "Po usunięciu obserwacji Cooka")

            if acc_clean >= val_acc:
                X_train_final = X_train_clean
                y_train_final = y_train_clean
                print(f"✅ [ZATWIERDZONO: Usunięto obserwacje wpływowe. Nowe Baseline Accuracy: {acc_clean:.4f}]")
                val_acc = acc_clean
            else:
                print(f"❌ [ODRZUCONO: Usunięcie pogorszyło model. Zostajemy przy starym zbiorze. Accuracy: {val_acc:.4f}]")
        else:
            print("--- Brak obserwacji drastycznie wpływowych.")
    except Exception as e:
        print(f"⚠️ [POMINIĘTO Cook'a: Logit zwrócił błąd: {e}]")

    print("\n" + "=" * 55)
    print(f"🎉 ETAP 1 UKOŃCZONY. Ostateczna dokładność walidacji K-NN: {val_acc:.4f}")
    print(f"Końcowe cechy: {X_train_final.columns.tolist()}")
    print("=" * 55)
