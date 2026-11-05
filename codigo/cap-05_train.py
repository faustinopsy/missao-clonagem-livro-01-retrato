# cap-05_train.py
# Missão: treinar um classificador k-NN com as SUAS amostras e medir se ele aprendeu.

import csv                # ler o arquivo de amostras

import joblib             # salvar o modelo treinado em arquivo
import numpy as np        # NumPy: tabelas de números
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_FILE = "dados-pessoais/expressoes/amostras.csv"
MODEL_FILE = "dados-pessoais/expressoes/modelo_knn.joblib"
INFO_COLUMNS = ["pessoa", "origem", "classe"]   # colunas que NÃO são características

TEST_SIZE = 0.2       # 20% das amostras ficam guardadas para a "prova"
NEIGHBORS = 5         # k: quantos vizinhos votam
RANDOM_SEED = 42      # semente: faz a separação sair igual toda vez


def load_samples(path):
    """Lê o CSV e devolve: nomes das características, características (X) e classes (y)."""
    features = []
    labels = []
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        feature_names = []
        for name in reader.fieldnames:
            if name not in INFO_COLUMNS:
                feature_names.append(name)
        for row in reader:
            values = []
            for name in feature_names:
                values.append(float(row[name]))
            features.append(values)
            labels.append(row["classe"])
    return feature_names, np.array(features), np.array(labels)


feature_names, X, y = load_samples(DATA_FILE)
classes = sorted(set(y))
print(f"Amostras: {len(X)} | características por amostra: {X.shape[1]}")
for class_name in classes:
    print(f"  {class_name:10} {np.sum(y == class_name)} amostras")

# 1. Separa treino e teste (stratify mantém a mesma proporção de cada classe nos dois)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)
print(f"\nTreino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

# 2. Treina o k-NN
model = KNeighborsClassifier(n_neighbors=NEIGHBORS)
model.fit(X_train, y_train)

# 3. Avalia: nota no treino (o que ele já viu) e no teste (o que ele nunca viu)
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"\nAcurácia no treino: {train_accuracy:.3f}")
print(f"Acurácia no teste:  {test_accuracy:.3f}")

# 4. Matriz de confusão: linha = classe verdadeira, coluna = palpite do modelo
matrix = confusion_matrix(y_test, model.predict(X_test), labels=classes)
print("\nMatriz de confusão (linha = verdadeiro, coluna = palpite):")
header = " " * 12
for class_name in classes:
    header += f"{class_name:>10}"
print(header)
for class_name, line in zip(classes, matrix):
    text = f"{class_name:12}"
    for value in line:
        text += f"{value:>10}"
    print(text)

# 5. Salva o modelo e a ordem das características (precisamos dela ao vivo)
joblib.dump({"model": model, "features": feature_names}, MODEL_FILE)
print(f"\nModelo salvo em {MODEL_FILE}")
