import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Konfiguracja strony
st.set_page_config(
    page_title="Wirtualny Sommelier",
    layout="centered"
)

# Niestandardowy styl CSS (motyw głębokiej czerwieni/burgundu wina i złota)
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        color: #D4AF37;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #A0A0A0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #722F37 !important;
        color: white !important;
        border-radius: 8px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        padding: 0.6rem 2rem !important;
        border: none !important;
        width: 100% !important;
        transition: background-color 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #8A3A43 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Wirtualny Sommelier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Wprowadź laboratoryjne cechy fizykochemiczne wina, a model sztucznej inteligencji (Random Forest) oceni jego jakość.</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'wine_model_light.pkl')
    return joblib.load(model_path)

try:
    model_pipeline, required_features = load_model()
except FileNotFoundError:
    st.error("Nie znaleziono pliku modelu 'wine_model_light.pkl'. Uruchom najpierw skrypt 'etap4_eksport.py'!")
    st.stop()

# Słownik definicji suwaków (nazwa_cechy: (etykieta, min, max, rozdzielczość, wartość_domyślna, pomoc))
slider_defs = {
    'alcohol': ("Zawartość alkoholu (%)", 8.0, 15.0, 0.1, 10.5, "Zawartość procentowa alkoholu w badanej próbce wina."),
    'sulphates': ("Siarczany (g/dm³)", 0.3, 2.0, 0.01, 0.62, "Ilość siarczanów, które działają jako przeciwutleniacz i środek konserwujący."),
    'volatile acidity': ("Kwasowość lotna (g/dm³)", 0.1, 1.6, 0.01, 0.52, "Ilość kwasu octowego w winie. Zbyt wysoki poziom daje nieprzyjemny ocetowy posmak."),
    'total sulfur dioxide': ("Całkowity dwutlenek siarki (mg/dm³)", 6.0, 200.0, 1.0, 46.0, "Całkowita ilość wolnych i związanych form dwutlenku siarki (SO2)."),
    'density': ("Gęstość wina (g/cm³)", 0.9900, 1.0040, 0.0001, 0.9967, "Gęstość wina zbliżona do gęstości wody, zależna od cukru i alkoholu.")
}

st.markdown("### Parametry laboratoryjne wina:")

user_inputs = {}

# Generowanie suwaków w estetycznych sekcjach
for feature in required_features:
    label, min_val, max_val, res, default_val, help_text = slider_defs[feature]
    user_inputs[feature] = st.slider(
        label=label,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=res,
        help=help_text
    )

st.markdown("---")

# Predykcja
if st.button("Zbadaj Jakość Wina"):
    input_df = pd.DataFrame([user_inputs])
    
    prediction = model_pipeline.predict(input_df)[0]
    probabilities = model_pipeline.predict_proba(input_df)[0]
    prob_percent = probabilities[prediction] * 100
    
    st.markdown("### Wynik Oceny Sommeliera:")
    
    if prediction == 1:
        st.balloons()
        st.success("### **Wino Dobre / Premium**")
        st.info(f"Model sztucznej inteligencji prognozuje klasę **PREMIUM** z prawdopodobieństwem **{prob_percent:.1f}%**.")
    else:
        st.error("### **Wino Przeciętne / Słabe**")
        st.warning(f"Model sztucznej inteligencji prognozuje klasę **PRZECIĘTNĄ** z prawdopodobieństwem **{prob_percent:.1f}%**.")

st.caption("Projekt końcowy: Uczenie Maszynowe II — Wirtualny Sommelier")
