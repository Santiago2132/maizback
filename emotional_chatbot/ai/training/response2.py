import os
import json
import joblib
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar lematizador y stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Cargar modelos y recursos
rf_model = joblib.load("ai/models/emotion_rf_model.pkl")
mlp_model = tf.keras.models.load_model("ai/models/emotion_mlp_model.h5")
vectorizer = joblib.load("ai/models/vectorizer.pkl")
label_classes = np.load("ai/models/label_encoder_classes.npy", allow_pickle=True)

# Cargar respuestas predefinidas desde múltiples archivos JSON
response_map = {}
intents_files = ["ai/data/intents.json", "ai/data/extra_intents.json"]

for file in intents_files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            intents = json.load(f)
        for intent in intents['intents']:
            response_map[intent['tag']] = intent['responses']

def preprocess_text(text):
    """Preprocesa el texto eliminando stopwords y aplicando lematización."""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

def predict_emotion(text):
    """ Predice la emoción de un texto utilizando ambos modelos. """
    preprocessed_text = preprocess_text(text)
    X_input = vectorizer.transform([preprocessed_text]).toarray()
    rf_prediction = rf_model.predict(X_input)[0]
    mlp_prediction = np.argmax(mlp_model.predict(X_input), axis=1)[0]
    
    # Tomar la predicción más frecuente
    final_prediction = rf_prediction if rf_prediction == mlp_prediction else mlp_prediction
    return label_classes[final_prediction]

def generate_response(user_input):
    """ Genera una respuesta basada en la emoción detectada. """
    emotion = predict_emotion(user_input)
    
    if emotion in response_map:
        return np.random.choice(response_map[emotion])  # Escoge una respuesta aleatoria de la lista
    
    return "Lo siento, no entendí tu emoción. ¿Puedes decirlo de otra manera?"

if __name__ == "__main__":
    while True:
        user_text = input("Tú: ")
        if user_text.lower() in ["salir", "exit", "quit"]:
            break
        print("Bot:", generate_response(user_text))
