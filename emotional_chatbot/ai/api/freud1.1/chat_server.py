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
from flask import Flask, request, jsonify

# Descargamos recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración inicial
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("spanish"))
DEBUG = True  # Cambiar a False para desactivar mensajes de depuración

# Función para cargar modelos
def load_models():
    models = {
        'mlp_model': tf.keras.models.load_model("ai/models/emotion_mlp_model.keras"),
        'vectorizer': joblib.load("ai/models/vectorizer.pkl"),
        'label_encoder': np.load("ai/models/label_encoder_classes.npy", allow_pickle=True)
    }
    le = LabelEncoder()
    le.classes_ = models['label_encoder']
    models['label_encoder'] = le
    return models

# Función para cargar datos de intenciones
def load_intents():
    response_map = {}
    patterns_by_intent = {}
    intents_files = ["ai/data/intents.json", "ai/data/extra_intents.json"]
    
    for file in intents_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for intent in data['intents']:
                tag = intent['tag']
                if tag not in response_map:
                    response_map[tag] = []
                    patterns_by_intent[tag] = []
                response_map[tag].extend(intent['responses'])
                patterns_by_intent[tag].extend(intent['patterns'])
    
    if DEBUG:
        print("\nIntenciones cargadas:")
        for tag, patterns in patterns_by_intent.items():
            print(f"Tag: {tag} | Patrones: {patterns[:2]}... ({len(patterns)} patrones)")
    
    return response_map, patterns_by_intent

# Función para cargar palabras ofensivas
def load_offensive_words():
    offensive_words = []
    if os.path.exists("ai/data/dictionary_word_dataset.csv"):
        df = pd.read_csv("ai/data/dictionary_word_dataset.csv", header=None, encoding="utf-8")
        offensive_words = [lemmatizer.lemmatize(word.lower()) for word in df[0].astype(str).tolist()]
    return offensive_words

# Preprocesamiento de texto
def text_preprocessing(text):
    text = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s]', '', text)
    text = text.lower().strip()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Detección de similitud
def is_similar_to_intent(input_text, patterns, threshold=65):
    processed_input = text_preprocessing(input_text)
    for pattern in patterns:
        processed_pattern = text_preprocessing(pattern)
        similarity = fuzz.token_sort_ratio(processed_input, processed_pattern)
        if DEBUG:
            print(f"Comparando: '{input_text}' vs '{pattern}' → {similarity}%")
        if similarity > threshold:
            return True
    return False

# Predicción de emoción
def predict_emotion(text, models):
    preprocessed_text = text_preprocessing(text)
    X_input = models['vectorizer'].transform([preprocessed_text])
    mlp_probs = models['mlp_model'].predict(X_input.toarray(), verbose=0)[0]
    predicted_class_idx = np.argmax(mlp_probs)
    confidence = mlp_probs[predicted_class_idx]
    
    if confidence < 0.6:  # Umbral de confianza
        return "unknown"
    
    return models['label_encoder'].classes_[predicted_class_idx]

# Detección de palabras ofensivas
def detect_offensive_words(text, offensive_words):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return any(word in offensive_words for word in tokens)

# Generación de respuesta
def generate_response(user_input, models, response_map, patterns_by_intent, offensive_words):
    # Detección de palabras ofensivas
    if detect_offensive_words(user_input, offensive_words):
        return "Lo siento, pero mantengamos nuestra conversación respetuosa."
    
    # Búsqueda por coincidencia de patrones
    for tag, patterns in patterns_by_intent.items():
        if is_similar_to_intent(user_input, patterns, 65):
            if DEBUG:
                print(f"Coincidencia por patrón con tag: {tag}")
            return np.random.choice(response_map[tag])
    
    # Predicción por modelo ML
    emotion = predict_emotion(user_input, models)
    if DEBUG:
        print(f"Predicción del modelo: {emotion}")
    
    if emotion != "unknown" and emotion in response_map:
        return np.random.choice(response_map[emotion])
    
    return "No estoy seguro de qué decirte... ¿Puedes darme más detalles?"

# Inicialización del sistema
def initialize_system():
    models = load_models()
    offensive_words = load_offensive_words()
    response_map, patterns_by_intent = load_intents()
    return models, offensive_words, response_map, patterns_by_intent

# Configuración del servidor Flask
app = Flask(__name__)

# Cargar modelos y datos al iniciar el servidor
models, offensive_words, response_map, patterns_by_intent = initialize_system()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "No se proporcionó un mensaje"}), 400
        
        bot_response = generate_response(
            user_input=user_message,
            models=models,
            response_map=response_map,
            patterns_by_intent=patterns_by_intent,
            offensive_words=offensive_words
        )
        
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)