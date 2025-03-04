import os
import json
import joblib
import numpy as np
import tensorflow as tf
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import fuzz

# Descargamos recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializamos lematizador y stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("spanish"))

# Cargamos modelos y recursos
mlp_model = tf.keras.models.load_model("ai/models/emotion_mlp_model.keras")
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy')
vectorizer = joblib.load("ai/models/vectorizer.pkl")
label_classes = np.load("ai/models/label_encoder_classes.npy", allow_pickle=True)

# Reconstruimos el LabelEncoder
le = LabelEncoder()
le.classes_ = label_classes

# Cargamos respuestas predefinidas
response_map = {}
intents_files = ["ai/data/intents.json", "ai/data/extra_intents.json"]

for file in intents_files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            intents = json.load(f)
        for intent in intents['intents']:
            response_map[intent['tag']] = intent['responses']

# Extraemos patrones para cada intención
patterns_by_intent = {intent['tag']: intent['patterns'] for intent in intents['intents']}

# Función para verificar similitud con patrones
def is_similar_to_intent(input_text, patterns, threshold=70):
    """Verifica si el texto es similar a algún patrón de la intención."""
    for pattern in patterns:
        if fuzz.token_sort_ratio(input_text.lower(), pattern.lower()) > threshold:
            return True
    return False

# Función para cargar palabras ofensivas
def load_offensive_words(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None, encoding="utf-8")
        return [lemmatizer.lemmatize(word.lower()) for word in df[0].astype(str).tolist()]
    return []

offensive_words = load_offensive_words("ai/data/dictionary_word_dataset.csv")

def detect_offensive_words(text, offensive_words):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return [word for word in tokens if word in offensive_words]

def text_preprocessing(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    text = text.lower().strip()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def predict_emotion(text):
    preprocessed_text = text_preprocessing(text)
    X_input = vectorizer.transform([preprocessed_text])
    mlp_probs = mlp_model.predict(X_input.toarray(), verbose=0)[0]
    predicted_class_idx = np.argmax(mlp_probs)
    predicted_class = le.classes_[predicted_class_idx]
    confidence = mlp_probs[predicted_class_idx]
    return predicted_class if confidence > 0.5 else "unknown"

def generate_response(user_input):
    """Genera una respuesta combinando similitud y predicción."""
    # Verifica palabras ofensivas
    if detect_offensive_words(user_input, offensive_words):
        return "Lo siento, pero mantengamos nuestra conversación respetuosa."
    
    # Verifica similitud con intenciones definidas
    for intent, patterns in patterns_by_intent.items():
        if is_similar_to_intent(user_input, patterns):
            return np.random.choice(response_map[intent])
    
    # Si no hay coincidencia, intenta predecir la emoción
    emotion = predict_emotion(user_input)
    if emotion == "unknown" or emotion not in response_map:
        return "No estoy seguro de qué decirte... ¿Puedes darme más detalles?"
    return np.random.choice(response_map[emotion])

if __name__ == "__main__":
    while True:
        user_text = input("Tú: ")
        if user_text.lower() in ["exit", "quit"]:
            print("Bot: ¡Adiós!")
            break
        response = generate_response(user_text)
        print(f"Bot: {response}")