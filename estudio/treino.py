# treino.py — treinar, avaliar e salvar o modelo do Estúdio.
#
# Mesma receita do Capítulo 5 (k-NN), agora em funções que a janela chama,
# mais a avaliação honesta: "deixando uma pessoa de fora".

import csv
import time

import joblib                     # salvar e carregar o modelo em arquivo
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

MODEL_FILE = "modelo.joblib"
INFO_COLUMNS = ["pessoa", "origem", "classe"]   # colunas que NÃO são características
RANDOM_SEED = 42                                # a separação sai igual toda vez


def load_samples(csv_path):
    """Lê o amostras.csv. Devolve: nomes das características, X, y e a pessoa de cada amostra."""
    features = []
    labels = []
    people = []
    with open(csv_path, newline="", encoding="utf-8") as file:
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
            people.append(row["pessoa"])
    return feature_names, np.array(features), np.array(labels), np.array(people)


def check_data(y, neighbors):
    """Confere se dá para treinar. Se não der, levanta um erro com a explicação."""
    classes = sorted(set(y))
    if len(classes) < 2:
        raise ValueError("É preciso ter pelo menos 2 classes com amostras.")
    for name in classes:
        if np.sum(y == name) < 2:
            raise ValueError(f"A classe '{name}' tem só 1 amostra. Grave pelo menos 2.")
    if neighbors >= len(y):
        raise ValueError(f"k = {neighbors} é grande demais para {len(y)} amostras.")


def split_and_evaluate(X, y, neighbors, test_percent):
    """Separação aleatória em treino e teste (como no Cap. 5): treina, mede e monta a matriz."""
    classes = sorted(set(y))
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_percent / 100, stratify=y, random_state=RANDOM_SEED
        )
    except ValueError:
        raise ValueError(f"Com {test_percent}% para teste não sobra 1 amostra de cada classe "
                         "para a prova. Aumente o % de teste ou grave mais amostras.")
    if neighbors > len(X_train):
        raise ValueError(f"k = {neighbors} é maior que as {len(X_train)} amostras de treino.")
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(X_train, y_train)
    guesses = model.predict(X_test)
    return {
        "classes": classes,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, guesses),
        "matrix": confusion_matrix(y_test, guesses, labels=classes),
    }


def leave_one_person_out(X, y, people, neighbors):
    """Para cada pessoa: treina SEM ela e testa SÓ nela. É a prova mais honesta:
    o modelo nunca viu aquele rosto, como vai acontecer com um usuário novo."""
    results = []
    for person in sorted(set(people)):
        is_person = people == person              # True nas amostras dessa pessoa
        X_train, y_train = X[~is_person], y[~is_person]   # ~ inverte: todas as OUTRAS
        X_test, y_test = X[is_person], y[is_person]
        k = min(neighbors, len(X_train))          # poucas amostras de treino? k diminui
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        guesses = model.predict(X_test)
        results.append({
            "person": person,
            "samples": len(X_test),
            "hits": int(np.sum(guesses == y_test)),
            "accuracy": accuracy_score(y_test, guesses),
            "k": k,
        })
    return results


def train_final(X, y, neighbors):
    """Depois de avaliar, treina de novo com TODAS as amostras: é esse modelo que vai ser usado."""
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(X, y)
    return model


def save_model(path, model, feature_names, info):
    """Guarda o modelo, a ordem das características e um resumo do treino."""
    info["saved_at"] = time.strftime("%Y-%m-%d %H:%M")
    classes = []
    for name in model.classes_:
        classes.append(str(name))     # str(): texto comum, não np.str_ (lembra do Cap. 5?)
    joblib.dump({"model": model, "features": feature_names, "classes": classes, "info": info}, path)


def load_model(path):
    """Lê o arquivo salvo por save_model (usado na aba Testar)."""
    return joblib.load(path)
