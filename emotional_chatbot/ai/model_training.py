import os
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer#type: ignore
from sklearn.ensemble import RandomForestClassifier#type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.preprocessing import LabelEncoder#type: ignore
import joblib

def load_and_preprocess_data():
    """Carga y preprocesa los datos desde archivos CSV y JSON."""
    emotions_path = "ai/data/emotional_dataset.csv"
    intents_path = "ai/data/intents.json"
    
    if not os.path.exists(emotions_path):
        raise FileNotFoundError(f"Archivo no encontrado: {emotions_path}")
    if not os.path.exists(intents_path):
        raise FileNotFoundError(f"Archivo no encontrado: {intents_path}")
    
    emotions_df = pd.read_csv(emotions_path)
    with open(intents_path, encoding='utf-8') as f:
        intents = json.load(f)
    
    texts = emotions_df['text'].tolist() + [pattern for intent in intents['intents'] for pattern in intent['patterns']]
    labels = emotions_df['emotion'].tolist() + [intent['tag'] for intent in intents['intents'] for _ in intent['patterns']]
    
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # Filtrar clases con menos de 2 muestras
    unique, counts = np.unique(encoded_labels, return_counts=True)
    valid_classes = {cls for cls, count in zip(unique, counts) if count > 1}
    filtered_indices = [i for i, lbl in enumerate(encoded_labels) if lbl in valid_classes]
    
    if len(filtered_indices) < len(encoded_labels):
        print(f"⚠️ Eliminando {len(encoded_labels) - len(filtered_indices)} muestras de clases con menos de 2 instancias.")
    
    texts = [texts[i] for i in filtered_indices]
    encoded_labels = [encoded_labels[i] for i in filtered_indices]
    
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=None)
    X = vectorizer.fit_transform(texts)
    
    return X, np.array(encoded_labels), vectorizer, le

def train_emotional_model():
    """Entrena un modelo de clasificación de emociones basado en texto."""
    X, y, vectorizer, le = load_and_preprocess_data()
    
    if len(set(y)) < 2:
        raise ValueError("El conjunto de datos debe tener al menos 2 clases para entrenar el modelo.")
    
    test_size = max(0.2, min(0.3, len(set(y)) / len(y)))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y if len(set(y)) > 1 else None)
    
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_val, y_val)
    print(f"✅ Modelo entrenado con Accuracy: {accuracy:.4f}")
    
    os.makedirs("ai/models", exist_ok=True)
    joblib.dump(model, "ai/models/emotion_model.pkl")
    joblib.dump(vectorizer, "ai/models/vectorizer.pkl")
    np.save("ai/models/label_encoder_classes.npy", le.classes_)
    
    return model

if __name__ == "__main__":
    train_emotional_model()
'''🐢SA'''