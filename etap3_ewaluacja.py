import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, auc
from sklearn.model_selection import learning_curve
import statsmodels.api as sm

from src.data_loader import load_and_split_wine_data

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(base_dir, 'data', 'winequality-red.csv'))

    print("Wczytywanie i przygotowywanie danych...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_wine_data(data_path)

    # Łączymy zbiór treningowy i walidacyjny
    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    numeric_features = X_train_val.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocesor z Etapu 1 (imputacja regresyjna -> log1p -> StandardScaler)
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42, max_iter=10)),
        ('log_transform', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    print("Detekcja i usuwanie punktów wpływowych metodą Cooka na połączonym zbiorze treningowym...")
    try:
        # Preprocesujemy dane tymczasowo, aby policzyć odległość Cooka
        X_train_val_prep = preprocessor.fit_transform(X_train_val)
        X_train_val_prep = pd.DataFrame(X_train_val_prep, columns=numeric_features)
        
        X_train_const = sm.add_constant(X_train_val_prep)
        logit_model = sm.Logit(y_train_val.values, X_train_const).fit(disp=0)
        influence = logit_model.get_influence()
        cooks_d = influence.cooks_distance[0]

        # Próg Cooka: 4 / N
        cook_threshold = 4.0 / len(X_train_val)
        safe_indices = np.where(cooks_d <= cook_threshold)[0]
        
        X_train_clean = X_train_val.iloc[safe_indices]
        y_train_clean = y_train_val.iloc[safe_indices]
        print(f"Usunięto {len(cooks_d) - len(safe_indices)} wpływowych obserwacji. Nowy rozmiar zbioru treningowego: {len(X_train_clean)}")
    except Exception as e:
        print(f"⚠️ [Logit Cook'a zwrócił błąd: {e}. Trenuję na pełnym zbiorze treningowym]")
        X_train_clean = X_train_val
        y_train_clean = y_train_val

    # Budujemy ostateczny zwycięski potok z dostrojonymi hiperparametrami
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42, n_jobs=-1)
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    print("\nTrenowanie ostatecznego potoku Lasu Losowego...")
    final_pipeline.fit(X_train_clean, y_train_clean)

    print("Ocena ostatecznego modelu na izolowanym zbiorze testowym...")
    y_pred_test = final_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    print("\n" + "=" * 55)
    print(f"🏆 OSTATECZNA DOKŁADNOŚĆ NA ZBIORZE TESTOWYM: {test_acc:.4f}")
    print("=" * 55)

    print("\nRAPORT KLASYFIKACJI (ZBIÓR TESTOWY):")
    print(classification_report(y_test, y_pred_test, target_names=["Przeciętne/Słabe (0)", "Dobre/Premium (1)"]))

    # Tworzenie i zapisywanie macierzy pomyłek
    print("Generowanie macierzy pomyłek...")
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Przeciętne/Słabe (0)", "Dobre/Premium (1)"])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Macierz Pomyłek - Zbiór Testowy (Klasyfikacja Binarna)')
    
    plot_path = os.path.join(base_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Macierz pomyłek zapisana pomyślnie pod ścieżką: {plot_path}")
    
    # Tworzenie i zapisywanie krzywej ROC
    print("Generowanie krzywej ROC...")
    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    ax_roc.plot(fpr, tpr, color='#722F37', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='#A0A0A0', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('Odsetek fałszywie dodatnich (FPR)')
    ax_roc.set_ylabel('Odsetek prawdziwie dodatnich (TPR)')
    ax_roc.set_title('Krzywa ROC - Zbiór Testowy (Klasyfikacja Binarna)')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle=':', alpha=0.6)
    
    roc_path = os.path.join(base_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Krzywa ROC zapisana pomyślnie pod ścieżką: {roc_path}")
    
    # Tworzenie i zapisywanie krzywych uczenia (Learning Curves)
    print("Generowanie krzywych uczenia...")
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=final_pipeline,
        X=X_train_clean,
        y=y_train_clean,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig_lc, ax_lc = plt.subplots(figsize=(8, 6))
    ax_lc.plot(train_sizes, train_mean, 'o-', color='#722F37', label='Dokładność treningowa')
    ax_lc.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#722F37')
    ax_lc.plot(train_sizes, val_mean, 's-', color='#1f77b4', label='Dokładność walidacyjna (CV)')
    ax_lc.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#1f77b4')
    
    ax_lc.set_xlabel('Rozmiar zbioru treningowego')
    ax_lc.set_ylabel('Dokładność (Accuracy)')
    ax_lc.set_title('Krzywe Uczenia - Random Forest (Zbiór Treningowy)')
    ax_lc.legend(loc='lower right')
    ax_lc.grid(True, linestyle=':', alpha=0.6)
    
    lc_path = os.path.join(base_dir, 'learning_curve.png')
    plt.savefig(lc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Krzywe uczenia zapisane pomyślnie pod ścieżką: {lc_path}")
