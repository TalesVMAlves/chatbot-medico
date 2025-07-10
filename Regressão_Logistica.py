import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

X_val = validation_df.drop(columns=['TIPO']).values
y_val = validation_df['TIPO']

X_test = test_df.drop(columns=['TIPO']).values
y_test = test_df['TIPO']

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print("Relatório de Classificação Regressão Logística:")
print(classification_report(y_val, y_pred))

print("Matriz de Confusão Regressão Logística:")
conf_mat = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão Regressão Logística - Validação')
plt.show()
