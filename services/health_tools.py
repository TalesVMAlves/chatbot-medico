from langchain_chroma import Chroma
import joblib
import numpy as np
import pandas as pd
from typing import Dict
import re
from utils.get_embedding_function import get_embedding_function
from classes import Sintomas
import shap
import matplotlib.pyplot as plt

modelo = joblib.load('artefatos/lr_classificador_respiratorio.pkl')
explainer = joblib.load('artefatos/shap_explainer.pkl')
shap_values = joblib.load('artefatos/shap_values.pkl')

def query_rag(query_text: str) -> str:
    embedding_function = get_embedding_function()

    db = Chroma(persist_directory="knowledge_base_chroma", embedding_function=embedding_function)

    results = db.similarity_search_with_score(f"{query_text}", k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text

def lr_classifier(sintomas: Sintomas) -> str:
    X_exemplo = pd.DataFrame([sintomas.dict()])

    probs = modelo.predict_proba(X_exemplo)
    classe_predita_idx = np.argmax(probs[0])
    classe_predita = modelo.classes_[classe_predita_idx]
    probabilidade = probs[0][classe_predita_idx] * 100
    
    shap_values_instance = explainer(X_exemplo)

    shap.force_plot(
        base_value=explainer.expected_value[classe_predita_idx],
        shap_values=shap_values_instance.values[:, :, classe_predita_idx],
        features=X_exemplo,
        matplotlib=True,
        show=False 
    )
    plt.savefig("artefatos/shap_force.png", bbox_inches='tight')
    plt.close()
    
    return f"Classe predita: {classe_predita} ({probabilidade:.2f}%)"

## Não é o suficiente para Liguagem Natural, utilizar a função que chama uma LLM e preenche a classe Pydantic.

# sintomas_padronizados = {s.lower().replace("_", " "): s for s in Sintomas.__fields__}
# def extrair_sintomas(texto: str) -> Sintomas:
#     texto = texto.lower()
#     sintomas_extraidos: Dict[str, bool] = {sintoma: False for sintoma in Sintomas.__fields__}
    
#     for chave_normalizada, campo in sintomas_padronizados.items():
#         if re.search(rf'\b{re.escape(chave_normalizada)}\b', texto):
#             sintomas_extraidos[campo] = True
#     return Sintomas(**sintomas_extraidos)