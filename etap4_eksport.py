import os
import warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_and_split_wine_data

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(base_dir, 'data', 'winequality-red.csv'))

    print("Wczytywanie i przygotowywanie pełnych danych...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_wine_data(data_path)

    # Łączymy wszystkie zbiory, aby obliczyć ważność cech na pełnym zbiorze danych
    X_full = pd.concat([X_train, X_val, X_test], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val, y_test], axis=0).reset_index(drop=True)

    numeric_features = X_full.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocesor z Etapu 1
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42, max_iter=10)),
        ('log_transform', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    clf = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42, n_jobs=-1)
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    print("Trenowanie modelu na pełnym zbiorze danych w celu obliczenia ważności cech...")
    full_pipeline.fit(X_full, y_full)

    # Odczytujemy ważność cech
    importances = full_pipeline.named_steps['classifier'].feature_importances_
    
    fi_df = pd.DataFrame({
        'Cecha': numeric_features,
        'Znaczenie': importances
    }).sort_values(by='Znaczenie', ascending=False)

    print("\n" + "-" * 55)
    print("--- TOP 5 NAJWAŻNIEJSZYCH CECH (Wnioski Fizykochemiczne) ---")
    print("-" * 55)
    print(fi_df.head(5).to_string(index=False))
    print("-" * 55)

    # Wybieramy top 5 najważniejszych cech bazowych dla wdrożenia
    top_5_features = fi_df['Cecha'].head(5).tolist()
    print(f"\nWybrane cechy bazowe do aplikacji okienkowej: {top_5_features}")

    X_light = X_full[top_5_features]

    # Uczymy lekki model (Lightweight RF) na 5 cechach do wdrożenia
    light_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42, n_jobs=-1))
    ])

    print("\nTrenowanie lekkiego modelu (Lightweight RF) dla aplikacji...")
    light_pipeline.fit(X_light, y_full)

    # Zapisujemy model do pliku .pkl
    model_path = os.path.join(base_dir, 'wine_model_light.pkl')
    joblib.dump((light_pipeline, top_5_features), model_path)
    print(f"Lekki model wyeksportowany pomyślnie do pliku: {model_path}")
