import os
import json
import joblib
import numpy as np

# Rutas del modelo
MODEL_PATH = "ai/models2/emotion_model.pipeline"
ENCODER_PATH = "ai/models2/label_encoder.pkl"
INTENTS_PATH = "ai/data/intents.json"

# Cargar respuestas predefinidas
if not os.path.exists(INTENTS_PATH):
    raise FileNotFoundError("❌ No se encontró el archivo de intents.json.")

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

response_map = {intent['tag']: intent['responses'] for intent in intents['intents']}

def load_model():
    """Carga el modelo entrenado y el codificador de etiquetas."""
    if not all(os.path.exists(path) for path in [MODEL_PATH, ENCODER_PATH]):
        raise FileNotFoundError("❌ No se encontraron los archivos del modelo. Entrena el modelo primero.")
    
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder

def predict_emotion(text):
    """ Predice la emoción de un texto. """
    model, label_encoder = load_model()
    prediction = model.predict([text])[0]
    emotion = label_encoder.inverse_transform([prediction])[0]
    return emotion

def generate_response(user_input):
    """ Genera una respuesta basada en la emoción detectada. """
    emotion = predict_emotion(user_input)
    
    if emotion in response_map:
        return np.random.choice(response_map[emotion])  # Escoge una respuesta aleatoria
    
    return "Lo siento, no entendí tu emoción. ¿Puedes decirlo de otra manera?"

if __name__ == "__main__":
    while True:
        user_text = input("Tú: ")
        if user_text.lower() in ["salir", "exit", "quit"]:
            break
        print("Bot:", generate_response(user_text))
