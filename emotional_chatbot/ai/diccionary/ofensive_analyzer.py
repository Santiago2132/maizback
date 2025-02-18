<<<<<<< HEAD
import tensorflow as tf
from tensorflow import keras#type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer#type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences#type: ignore
import numpy as np
=======
import os
import re
import nltk
>>>>>>> patricia
import pandas as pd

# Descargar el recurso necesario para tokenización
nltk.download('punkt')

# Ruta del archivo CSV
offensive_path = "../data/dictionary_word_dataset.csv"

# Verificar si el archivo existe (comentado para que no imprima)
# def check_file_exists(file_path):
#     return os.path.exists(file_path)

# Cargar palabras ofensivas con pandas
def load_offensive_words_pandas(file_path):
    try:
        df = pd.read_csv(file_path, header=None, encoding="utf-8")
        words = df[0].astype(str).str.lower().tolist()
        return words
    except FileNotFoundError:
        return []
    except pd.errors.EmptyDataError:
        return []
    except pd.errors.ParserError:
        return []
    except Exception:
        return []

# Cargar palabras ofensivas manualmente con open()
def load_offensive_words_manual(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words = [line.strip().lower() for line in lines if line.strip()]
            return words
    except FileNotFoundError:
        return []
    except Exception:
        return []

# Función para tokenizar usando nltk o re.findall() como respaldo
def tokenize_text(text):
    try:
        return nltk.word_tokenize(text.lower())  # Convertir a minúsculas
    except Exception:
        return re.findall(r'\b\w+\b', text.lower())  # Método alternativo

# Función para detectar groserías en un texto
def detect_offensive_words(text, offensive_words):
    text = text.lower()  # Convertir a minúsculas
    words = tokenize_text(text)  # Obtener tokens
    print(f"🔎 Palabras detectadas en el texto: {words}")
    
    # Buscar coincidencias
    found_offensive_words = [word for word in words if word in offensive_words]

    if found_offensive_words:
        return f"🚫 Mensaje bloqueado. Palabras ofensivas detectadas: {', '.join(found_offensive_words)}"
    
    return "✅ Mensaje seguro"

# Cargar palabras ofensivas con la mejor opción disponible
# if check_file_exists(offensive_path):  # Comentado para evitar impresión
#     offensive_words = load_offensive_words_pandas(offensive_path)
#     if not offensive_words:
#         offensive_words = load_offensive_words_manual(offensive_path)
# else:
#     offensive_words = []

offensive_words = load_offensive_words_pandas(offensive_path)
if not offensive_words:
    offensive_words = load_offensive_words_manual(offensive_path)

# Modo interactivo para ingresar mensajes manualmente
while True:
    message = input("\nEscribe un mensaje (o escribe 'salir' para terminar): ")
    
    if message.lower() == "salir":
        print("👋 Programa finalizado.")
        break
    
    result = detect_offensive_words(message, offensive_words)
    print(f"Resultado: {result}")