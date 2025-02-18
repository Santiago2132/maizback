import os
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Dropout #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore

def load_and_preprocess_data():
    """Carga y preprocesa los datos desde múltiples diccionarios gratuitos."""
    datasets = [
        "ai/data/emotional_dataset.csv",
        "ai/data/sentiment_dataset.csv",
        "ai/data/expanded_emotion_data.csv"
    ]
    intents_paths = [
        "ai/data/intents.json",
        "ai/data/extra_intents.json"
    ]
    
    texts, labels = [], []
    le = LabelEncoder()
    
    for dataset in datasets:
        if os.path.exists(dataset):
            df = pd.read_csv(dataset)
            texts.extend(df['text'].tolist())
            labels.extend(df['emotion'].tolist())
    
    for intents_path in intents_paths:
        if os.path.exists(intents_path):
            with open(intents_path, encoding='utf-8') as f:
                intents = json.load(f)
            texts.extend([pattern for intent in intents['intents'] for pattern in intent['patterns']])
            labels.extend([intent['tag'] for intent in intents['intents'] for _ in intent['patterns']])
    
    encoded_labels = le.fit_transform(labels)
    
    # Filtrar clases con menos de 2 muestras
    unique, counts = np.unique(encoded_labels, return_counts=True)
    valid_classes = {cls for cls, count in zip(unique, counts) if count > 1}
    filtered_indices = [i for i, lbl in enumerate(encoded_labels) if lbl in valid_classes]
    
    texts = [texts[i] for i in filtered_indices]
    encoded_labels = [encoded_labels[i] for i in filtered_indices]
    
    vectorizer = TfidfVectorizer(max_features=15000, stop_words=None)
    X = vectorizer.fit_transform(texts).toarray()
    y = to_categorical(encoded_labels)
    
    return X, y, vectorizer, le

def train_emotional_model():
    """Entrena un modelo de clasificación de emociones basado en RandomForest y una red neuronal."""
    X, y, vectorizer, le = load_and_preprocess_data()
    
    if y.shape[1] < 2:
        raise ValueError("El conjunto de datos debe tener al menos 2 clases.")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y.argmax(axis=1))
    
    # Modelo 1: Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train.argmax(axis=1))
    rf_accuracy = rf_model.score(X_val, y_val.argmax(axis=1))
    print(f"✅ Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Modelo 2: Red Neuronal (MLP)
    mlp_model = Sequential([
        Dense(512, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(y.shape[1], activation='softmax')
    ])
    
    mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mlp_model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))
    mlp_accuracy = mlp_model.evaluate(X_val, y_val)[1]
    print(f"✅ MLP Neural Network Accuracy: {mlp_accuracy:.4f}")
    
    os.makedirs("ai/models", exist_ok=True)
    joblib.dump(rf_model, "ai/models/emotion_rf_model.pkl")
    joblib.dump(vectorizer, "ai/models/vectorizer.pkl")
    np.save("ai/models/label_encoder_classes.npy", le.classes_)
    mlp_model.save("ai/models/emotion_mlp_model.h5")
    
    return rf_model, mlp_model

if __name__ == "__main__":
    train_emotional_model()
