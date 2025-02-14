import os
import joblib
import numpy as np

# Rutas del modelo
MODEL_PATH = "M:/maizback/emotional_chatbotai/models2/emotion_model.pipeline"
ENCODER_PATH = "M:/maizback/emotional_chatbotai/models2/label_encoder.pkl"

def check_model_files():
    """Verifica que los archivos del modelo existen."""
    if not all(os.path.exists(path) for path in [MODEL_PATH, ENCODER_PATH]):
        raise FileNotFoundError("❌ No se encontraron los archivos del modelo. Entrena el modelo primero.")

def load_model():
    """Carga el modelo entrenado y el codificador de etiquetas."""
    check_model_files()
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder

def predict_emotion(text):
    """ Predice la emoción de un texto. """
    model, label_encoder = load_model()
    
    # Transformar el texto con el vectorizador del pipeline
    prediction = model.predict([text])[0]
    emotion = label_encoder.inverse_transform([prediction])[0]
    
    return emotion

if __name__ == "__main__":
    while True:
        user_text = input("📝 Ingrese un texto para analizar (o 'exit' para salir): ")
        if user_text.lower() in ["exit", "salir"]:
            break
        emotion = predict_emotion(user_text)
        print(f"😃 Emoción detectada: {emotion}")
