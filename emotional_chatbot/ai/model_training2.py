import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Configuración inicial de directorios
DIRECTORIES = ["ai/logs", "ai/models", "ai/metrics", "ai/data"]
for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai/logs/training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data():
    """Carga y preprocesa los datos con manejo correcto de clases"""
    try:
        logging.info("Iniciando carga y preprocesamiento de datos...")
        
        # Carga de datos
        emotions_df = pd.read_csv("ai/data/emotional_dataset.csv")
        with open("ai/data/intents.json", encoding='utf-8') as f:
            intents = json.load(f)
        
        # Procesamiento combinado
        texts = (
            emotions_df['text'].tolist() + 
            [pattern for intent in intents['intents'] for pattern in intent['patterns']]
        )
        labels = (
            emotions_df['emotion'].tolist() + 
            [intent['tag'] for intent in intents['intents'] for _ in intent['patterns']]
        )
        
        # Limpieza básica de texto
        texts = [str(t).strip().lower() for t in texts]
        
        # Codificación inicial de etiquetas
        le_original = LabelEncoder()
        encoded_labels = le_original.fit_transform(labels)
        
        # Filtrado de clases con menos de 5 muestras
        unique, counts = np.unique(encoded_labels, return_counts=True)
        valid_classes = unique[counts >= 5]
        mask = np.isin(encoded_labels, valid_classes)
        
        # Aplicar filtrado
        filtered_texts = [texts[i] for i, valid in enumerate(mask) if valid]
        filtered_labels = encoded_labels[mask]
        
        # Re-encodear etiquetas con solo las clases filtradas
        filtered_class_names = le_original.inverse_transform(filtered_labels)
        le_filtered = LabelEncoder()
        filtered_labels = le_filtered.fit_transform(filtered_class_names)
        
        # Vectorización
        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        X = vectorizer.fit_transform(filtered_texts)
        
        logging.info(
            f"Datos procesados: {X.shape[0]} muestras, "
            f"{len(le_filtered.classes_)} clases válidas"
        )
        
        return X, np.array(filtered_labels), vectorizer, le_filtered
    
    except Exception as e:
        logging.error(f"Error en preprocesamiento: {str(e)}")
        raise

def train_emotional_model():
    """Entrena el modelo con manejo correcto de las clases"""
    try:
        start_time = datetime.now()
        logging.info("Iniciando entrenamiento del modelo...")
        
        X, y, vectorizer, le = load_and_preprocess_data()
        
        # Validación de datos
        if len(np.unique(y)) < 2:
            raise ValueError("Se requieren al menos 2 clases para entrenamiento")
        
        # División estratificada
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.25,
            stratify=y,
            random_state=42
        )
        
        # Configuración del modelo
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        # Entrenamiento
        model.fit(X_train, y_train)
        
        # Predicción y evaluación
        y_pred = model.predict(X_val)
        accuracy = model.score(X_val, y_val)
        
        # Generar reporte con clases correctas
        unique_classes = np.unique(y_val)
        class_names = le.inverse_transform(unique_classes)
        
        report = classification_report(
            y_val,
            y_pred,
            labels=unique_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Métricas para guardar
        training_stats = {
            "timestamp": start_time.isoformat(),
            "training_time": str(datetime.now() - start_time),
            "accuracy": accuracy,
            "class_distribution": dict(zip(
                le.inverse_transform(np.unique(y)),
                np.bincount(y)
            )),
            "classification_report": report,
            "model_metadata": {
                "features": X.shape[1],
                "classes": le.inverse_transform(np.unique(y)).tolist(),
                "parameters": model.get_params()
            }
        }
        
        # Guardado de artefactos
        joblib.dump(model, "ai/models/emotion_model.pkl")
        joblib.dump(vectorizer, "ai/models/vectorizer.pkl")
        np.save("ai/models/label_encoder.npy", le.classes_)
        
        with open("ai/metrics/training_stats.json", "w") as f:
            json.dump(training_stats, f, indent=2, ensure_ascii=False)
        
        logging.info(
            f"Entrenamiento completado en {training_stats['training_time']}\n"
            f"Precisión final: {accuracy:.2%}\n"
            f"Clases detectadas: {len(le.classes_)}"
        )
        
        return training_stats
    
    except Exception as e:
        logging.error(f"Error en entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    train_emotional_model()