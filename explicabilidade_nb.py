from langchain_chroma import Chroma
import joblib
import numpy as np
import pandas as pd
from typing import Dict
import re
from classes import Sintomas
import shap
import matplotlib.pyplot as plt

modelo = joblib.load('artefatos/bnb_classificador_respiratorio.pkl')
explainer = joblib.load('artefatos/shap_explainer_nb.pkl')
shap_values = joblib.load('artefatos/shap_values_nb.pkl')

def bnb_classifier(sintomas: Sintomas) -> str:
    global modelo, explainer
    model = modelo
    explainer = explainer

    X_exemplo = pd.DataFrame([sintomas.dict()])

    probs = model.predict_proba(X_exemplo)
    classe_predita_idx = np.argmax(probs[0])
    classe_predita = model.classes_[classe_predita_idx]
    probabilidade = probs[0][classe_predita_idx] * 100

    shap_values_instance = explainer(X_exemplo)

    shap.force_plot(explainer.expected_value[1],
                shap_values_instance.values[0, :, classe_predita_idx],
                X_exemplo.iloc[0],
                link="logit")

    plt.savefig("artefatos/shap_force_nb.png", bbox_inches='tight')
    plt.close()

    return f"Classe predita: {classe_predita} ({probabilidade:.2f}%)"