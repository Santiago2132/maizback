import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

# Cargar dataset
df = pd.read_csv("C:/Users/Patty/Documents/proyecto/maizback/emotional_chatbot/ai/data/diccionary_word_dataset.csv")

# Asegurar que la columna de texto sea string
df['text'] = df['text'].astype(str)

# Crear etiquetas (1 para groserías, 0 para no groserías)
df['label'] = 1  # Asumimos que todas las palabras en el dataset son groserías

# Tokenización
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])

# Padding (para igualar longitudes)
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = np.array(df['label'])

# Crear modelo
def create_model(vocab_size, input_length):
    model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=16),  # Quita input_length
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Definir parámetros del modelo
vocab_size = len(tokenizer.word_index) + 1
model = create_model(vocab_size, max_length)

# Entrenar modelo
model.fit(X, y, epochs=10, verbose=1)

# Guardar el modelo
model_path = "/mnt/data/swear_word_model.keras"
model.save(model_path)

print(f"Modelo guardado en: {model_path}")