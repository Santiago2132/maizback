import os
import pandas as pd
import numpy as np
import json
import joblib
import re
import nltk
import tensorflow as tf  # Importar TensorFlow como tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

nltk.download(['punkt', 'stopwords', 'wordnet'])

def text_preprocessing(text):
    """Limpieza y normalización de texto avanzada"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Limpieza básica
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower().strip()
    
    # Tokenización y lematización
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

def load_and_preprocess_data():
    """Carga y preprocesa los datos con mejor manejo de texto"""
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
    
    # Carga de datos con preprocesamiento
    for dataset in datasets:
        if os.path.exists(dataset):
            df = pd.read_csv(dataset)
            df['text'] = df['text'].apply(text_preprocessing)
            texts.extend(df['text'].tolist())
            labels.extend(df['emotion'].tolist())
    
    for intents_path in intents_paths:
        if os.path.exists(intents_path):
            with open(intents_path, encoding='utf-8') as f:
                intents = json.load(f)
            for intent in intents['intents']:
                for pattern in intent['patterns']:
                    processed_pattern = text_preprocessing(pattern)
                    texts.append(processed_pattern)
                    labels.append(intent['tag'])
    
    # Filtrado de clases
    encoded_labels = le.fit_transform(labels)
    unique, counts = np.unique(encoded_labels, return_counts=True)
    valid_classes = {cls for cls, count in zip(unique, counts) if count >= 5}  # Mínimo 5 muestras
    
    filtered_indices = [i for i, lbl in enumerate(encoded_labels) if lbl in valid_classes]
    texts = [texts[i] for i in filtered_indices]
    encoded_labels = [encoded_labels[i] for i in filtered_indices]
    
    # Balanceo de clases con SMOTE
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    smote = SMOTE(k_neighbors=3, random_state=42)  # Ajuste de k_neighbors a 3
    X_balanced, y_balanced = smote.fit_resample(X, encoded_labels)
    
    y = to_categorical(y_balanced)
    
    return X_balanced, y, vectorizer, le

def train_emotional_model():
    """Modelo optimizado con mejor manejo de entrenamiento"""
    X, y, vectorizer, le = load_and_preprocess_data()
    
    if y.shape[1] < 2:
        raise ValueError("Se necesitan al menos 2 clases")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y.argmax(axis=1), random_state=42)
    
    # Optimización de Random Forest
    rf_params = {
        'n_estimators': 500,
        'max_depth': None,
        'min_samples_split': 5,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    }
    
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train.argmax(axis=1))
    rf_accuracy = rf_model.score(X_val, y_val.argmax(axis=1))
    print(f"✅ Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Mejorar modelo neuronal
    mlp_model = Sequential([
        tf.keras.Input(shape=(X.shape[1],)),  # Usar Input en lugar de input_shape
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(y.shape[1], activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    
    mlp_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    history = mlp_model.fit(
        X_train.toarray(), y_train,
        epochs=50,
        batch_size=128,
        validation_data=(X_val.toarray(), y_val),
        callbacks=[early_stop, reduce_lr]
    )
    
    mlp_accuracy = mlp_model.evaluate(X_val.toarray(), y_val)[1]
    print(f"✅ MLP Neural Network Accuracy: {mlp_accuracy:.4f}")
    
    # Guardar modelos
    os.makedirs("ai/models", exist_ok=True)
    joblib.dump(rf_model, "ai/models/emotion_rf_model.pkl")
    joblib.dump(vectorizer, "ai/models/vectorizer.pkl")
    np.save("ai/models/label_encoder_classes.npy", le.classes_)
    mlp_model.save("ai/models/emotion_mlp_model.keras")
    
    return rf_model, mlp_model

if __name__ == "__main__":
    train_emotional_model()
