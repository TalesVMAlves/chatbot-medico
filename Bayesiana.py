import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import numpy as np
import joblib

df = pd.read_csv(r'dataset\conjunto_balanceado.csv') 

train_df, temp_df = train_test_split(
    df,
    test_size=0.3, 
    random_state=42,
    stratify=df['TIPO']
)
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df['TIPO']
)

X_train = train_df.drop(columns=['TIPO'])
y_train = train_df['TIPO']

X_val = validation_df.drop(columns=['TIPO'])
y_val = validation_df['TIPO']

X_test = test_df.drop(columns=['TIPO']).values
y_test = test_df['TIPO']

model = BernoulliNB()
model.fit(X_train, y_train)
joblib.dump(model, 'artefatos/bnb_classificador_respiratorio.pkl')

explainer = shap.Explainer(model.predict_proba, X_train)
joblib.dump(explainer, 'artefatos/shap_explainer_nb.pkl')

shap_values = explainer(X_val)  
joblib.dump(shap_values, 'artefatos/shap_values_nb.pkl')
y_pred = model.predict(X_val)

print("Relatório de Classificação Bayesiana:")
print(classification_report(y_val, y_pred))

print("Matriz de Confusão Baysiana:")
conf_mat = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão Baysiana- Validação')
plt.show()

#Colocar sintomas manualmente
#sintomas_informados = []
#sintomas_maiuscula = [s.upper() for s in sintomas_informados]
#sintomas_formatado = [s.replace(' ', '_') for s in sintomas_maiuscula]
#colunas_modelo =  X_train.columns.tolist()


def preparar_entrada(sintomas_formatado, colunas_modelo):
    entrada = [0] * len(colunas_modelo)
    for sintoma in sintomas_formatado:
        if sintoma in colunas_modelo:
            idx = colunas_modelo.index(sintoma)
            entrada[idx] = 1
    return entrada

#Colocar sintomas no terminal
sintomas_usuario = input("Digite os sintomas separados por vírgula: ").split(',')
sintomas_maiuscula = [s.strip().upper() for s in sintomas_usuario]
sintomas_formatado = [s.replace(' ', '_') for s in sintomas_maiuscula]
colunas_modelo =  X_train.columns.tolist() 

# Preparar a entrada
entrada = preparar_entrada(sintomas_formatado, colunas_modelo)

# Checar se há sintomas válidos
if not any(entrada):
    print("Nenhum dos sintomas informados corresponde às colunas do modelo.")
else:
    probabilidades = model.predict_proba([entrada])[0]
    classes = model.classes_
    for classe, prob in zip(classes, probabilidades):
        print(f"{classe}: {prob:.2%}")
    print("Diagnóstico mais provável:", classes[probabilidades.argmax()])

