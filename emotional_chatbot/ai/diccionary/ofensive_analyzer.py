import os
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential#type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer#type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences#type: ignore
import numpy as np
import re
import pandas as pd
import pickle

# Descargar recursos de tokenización de nltk
nltk.download('punkt')

# Ruta del dataset
DATASET_PATH = "../data/dictionary_word_dataset.csv"
MODEL_PATH = "models/offensive_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# Tokenización con nltk
def tokenize_text(text):
    try:
        return nltk.word_tokenize(text.lower())  # Convertir a minúsculas
    except Exception:
        return re.findall(r'\b\w+\b', text.lower())  # Alternativa con regex

# Función para entrenar el modelo
def train_model():
    print("📥 Cargando dataset...")
    
    try:
        df = pd.read_csv(DATASET_PATH)  # Asegúrate de que el archivo existe y tiene datos
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en {DATASET_PATH}")
        return
    except pd.errors.EmptyDataError:
        print("❌ Error: El archivo CSV está vacío.")
        return
    except Exception as e:
        print(f"❌ Error inesperado al cargar el dataset: {e}")
        return

    print("🔍 Verificando estructura del dataset...")
    print(df.head())  # Muestra las primeras filas para verificar las columnas

    if "text" not in df.columns or "label" not in df.columns:
        print(f"❌ Error: El dataset debe contener las columnas 'text' y 'label'. Columnas encontradas: {df.columns}")
        return

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    print(df["label"].value_counts())

    print("🔄 Tokenizando texto...")
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

    X = np.array(padded_sequences)
    y = np.array(labels)

    print("🔧 Creando modelo...")
    model = Sequential([
        Embedding(10000, 16, input_length=100),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Clasificación binaria
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("🏋️ Entrenando modelo...")
    model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

    # Guardar modelo y tokenizer
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"✅ Modelo guardado en {MODEL_PATH}")
    print(f"✅ Tokenizer guardado en {TOKENIZER_PATH}")

# Cargar modelo entrenado
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("⚠️ No se encontró un modelo entrenado. Ejecuta train_model() primero.")
        return None, None

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    print("✅ Modelo y tokenizer cargados correctamente.")
    return model, tokenizer

# Detectar mensajes ofensivos
def detect_offensive_message(text, model, tokenizer):
    text = text.lower()
    words = tokenize_text(text)
    
    print(f"🔎 Palabras detectadas: {words}")
    
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    prediction = model.predict(padded)[0][0]

    print(f"📊 Probabilidad de mensaje ofensivo: {prediction}")
    return "🚫 Mensaje ofensivo" if prediction > 0.6396 else "✅ Mensaje seguro"

# Modo interactivo
def interactive_mode():
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        return

    while True:
        message = input("\nEscribe un mensaje (o 'salir' para terminar): ")
        if message.lower() == "salir":
            print("👋 Programa finalizado.")
            break
        result = detect_offensive_message(message, model, tokenizer)
        print(f"Resultado: {result}")

# Ejecutar el modo interactivo
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
    interactive_mode()