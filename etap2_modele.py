import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from src.data_loader import load_and_split_wine_data

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(base_dir, 'data', 'winequality-red.csv'))

    print("Wczytywanie danych...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_wine_data(data_path)

    # Łączymy zbiór treningowy i walidacyjny do 5-krotnej walidacji krzyżowej (CV)
    X_train_cv = pd.concat([X_train, X_val], axis=0)
    y_train_cv = pd.concat([y_train, y_val], axis=0)

    print(f"Rozmiar połączonego zbioru do Walidacji Krzyżowej: {len(X_train_cv)} próbek.")

    # Definiujemy cechy numeryczne
    numeric_features = X_train_cv.select_dtypes(include=[np.number]).columns.tolist()

    # Budujemy optymalny preprocesor na podstawie wyników Etapu 1 (imputacja regresyjna -> log1p -> scaler)
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42, max_iter=10)),
        ('log_transform', FunctionTransformer(np.log1p, validate=False)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Stosujemy StratifiedKFold ze względu na niezbalansowany zbiór (zapewnia równe proporcje klas we foldach)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Definiujemy zbiór zróżnicowanych algorytmów klasyfikacyjnych
    models = {
        "Naiwny Bayes": GaussianNB(),
        "Drzewo Decyzyjne": DecisionTreeClassifier(random_state=42),
        "Regresja Logistyczna": LogisticRegression(max_iter=1000, random_state=42),
        "K-NN (k=9)": KNeighborsClassifier(n_neighbors=9),
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
        "Las Losowy": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }

    print("\n" + "=" * 65)
    print(f"{'Model':<25} | {'Średni Acc (CV)':<16} | {'Odchylenie +/-'}")
    print("=" * 65)

    cv_results = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        try:
            scores = cross_val_score(pipe, X_train_cv, y_train_cv, cv=kf, scoring='accuracy', n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()
            cv_results[name] = mean_score

            print(f"{name:<25} | {mean_score:<16.4f} | {std_score:.4f}")
        except Exception as e:
            print(f"{name:<25} | BŁĄD OBLICZEŃ     | {str(e)[:30]}")

    print("=" * 65)

    best_model_name = max(cv_results, key=cv_results.get)
    print(f"\n🏆 ZWYCIĘZCA WALIDACJI KRZYŻOWEJ: {best_model_name} (Średnie Accuracy: {cv_results[best_model_name]:.4f})")
    
    if best_model_name == "Las Losowy":
        print("Model Lasu Losowego świetnie generalizuje i idealnie nadaje się do ekstrakcji cech biznesowych!")
        
        print("\n" + "=" * 65)
        print("🎯 ROZPOCZYNANIE STROJENIA HIPERPARAMETRÓW DLA LASU LOSOWEGO (GridSearchCV)...")
        print("=" * 65)
        
        # Definiujemy siatkę parametrów
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 15, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        # Tworzymy potok z domyślnym klasyfikatorem dla strojenia
        rf_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        
        # Uruchamiamy Grid Search z zachowaniem Stratified K-Fold
        grid_search = GridSearchCV(
            estimator=rf_pipe,
            param_grid=param_grid,
            cv=kf,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("Trwa dopasowywanie modeli na siatce parametrów...")
        grid_search.fit(X_train_cv, y_train_cv)
        
        print("\nWyniki strojenia:")
        print(f"Najlepsze parametry: {grid_search.best_params_}")
        print(f"Najlepszy wynik dokładności (CV): {grid_search.best_score_:.4f}")
        
        # Porównanie z domyślnym
        baseline_diff = grid_search.best_score_ - cv_results["Las Losowy"]
        if baseline_diff > 0:
            print(f"📈 Poprawa dokładności dzięki dostrojeniu: +{baseline_diff:.4f} (+{baseline_diff*100:.2f}%)")
        else:
            print("Model z domyślnymi parametrami okazał się optymalny.")
