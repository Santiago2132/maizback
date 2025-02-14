import json
import joblib
import os
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = "ai/data/conversation_trees.json"
MODEL_PATH = "ai/models/response_generator.pkl"

def load_training_data():
    """Carga los datos de entrenamiento desde el árbol de conversación."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("❌ No se encontró el archivo de conversaciones.")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = []
    responses = []

    def traverse(node):
        if node:
            messages.append(node["message"])
            if node["left"]:
                responses.append(node["left"]["message"])
                traverse(node["left"])
            if node["right"]:
                traverse(node["right"])

    traverse(data)

    return messages, responses

def train_response_model():
    """Entrena un modelo avanzado para generar respuestas más elaboradas."""
    messages, responses = load_training_data()

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X = vectorizer.fit_transform(messages)

    model = LogisticRegression()
    model.fit(X, responses)

    joblib.dump((model, vectorizer), MODEL_PATH)
    print("✅ Modelo de respuestas guardado.")

if __name__ == "__main__":
    train_response_model()
