import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
import numpy as np
import os
import re
import nltk
import pandas as pd

# Descargar el recurso necesario para tokenización
nltk.download('punkt')

# Ruta del archivo CSV
offensive_path = "../data/dictionary_word_dataset.csv"

# Cargar palabras ofensivas
def load_offensive_words(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, encoding="utf-8")  # Usar header=0 por defecto
            # Limpiar frases: quitar comillas, espacios extras, y normalizar
            offensive_phrases = df["text"].astype(str).str.lower().str.replace(r'[\'"\s]+', ' ', regex=True).str.strip().tolist()
            return offensive_phrases
        except Exception as e:
            print(f"Error: {e}")
            return []
    return []

# Tokenización
def tokenize_text(text):
    try:
        return nltk.word_tokenize(text.lower())
    except Exception:
        return re.findall(r'\b\w+\b', text.lower())

# Detección de palabras y frases ofensivas
def detect_offensive_words(text, offensive_words):
    # Normalizar el texto: eliminar caracteres especiales y espacios extras
    text_clean = re.sub(r'[^\w\s]', '', text.lower())  # Quitar puntuación
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()  # Unificar espacios
    
    # Buscar coincidencias exactas de frases
    found_phrases = [phrase for phrase in offensive_words if phrase in text_clean]
    
    return found_phrases

# Cargar palabras ofensivas
offensive_words = load_offensive_words(offensive_path)

# Preparar datos para el modelo
texts = ["Este es un mensaje seguro", "Este mensaje contiene groserías"]
labels = [0, 1]  # 0: seguro, 1: ofensivo

# Tokenización y padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=50)

# Crear y entrenar el modelo
model = keras.Sequential([
    keras.layers.Embedding(10000, 16, input_length=50),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, np.array(labels), epochs=10)

# Guardar el modelo
model_path = "M:/maizback/emotional_chatbot/ai/models"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)

# Modo interactivo
while True:
    message = input("\nEscribe un mensaje (o 'salir' para terminar): ")
    
    if message.lower() == "salir":
        print("👋 Programa finalizado.")
        break
    
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=50)
    prediction = model.predict(padded)
    
    if prediction[0] > 0.5:
        offensive_words_found = detect_offensive_words(message, offensive_words)
        print(f"🚫 Mensaje bloqueado. Palabras ofensivas detectadas: {', '.join(offensive_words_found)}")
    else:
        print("✅ Mensaje seguro")