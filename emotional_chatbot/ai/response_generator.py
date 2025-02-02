import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar modelos y recursos
model = joblib.load("ai/models/emotion_model.pkl")
vectorizer = joblib.load("ai/models/vectorizer.pkl")
label_classes = np.load("ai/models/label_encoder_classes.npy", allow_pickle=True)

# Cargar respuestas predefinidas desde intents.json
with open("ai/data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# Crear un diccionario de respuestas según la emoción detectada
response_map = {intent['tag']: intent['responses'] for intent in intents['intents']}

def predict_emotion(text):
    """ Predice la emoción de un texto y devuelve la etiqueta correspondiente. """
    X_input = vectorizer.transform([text])
    emotion_index = model.predict(X_input)[0]
    return label_classes[emotion_index]

def generate_response(user_input):
    """ Genera una respuesta basada en la emoción detectada. """
    emotion = predict_emotion(user_input)
    
    if emotion in response_map:
        return np.random.choice(response_map[emotion])  # Escoge una respuesta aleatoria de la lista
    
    return "Lo siento, no entendí tu emoción. ¿Puedes decirlo de otra manera?"
'''🐢'''
if __name__ == "__main__":
    while True:
        user_text = input("Tú: ")
        if user_text.lower() in ["salir", "exit", "quit"]:
            break
        print("Bot:", generate_response(user_text))
