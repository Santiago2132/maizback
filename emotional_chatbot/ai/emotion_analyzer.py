import joblib
import numpy as np
import os

MODEL_PATH = "ai/models/emotion_model.pkl"
VECTORIZER_PATH = "ai/models/vectorizer.pkl"
ENCODER_PATH = "ai/models/label_encoder_classes.npy"

def check_model_files():
    """Verifica que los archivos del modelo existen."""
    if not all(os.path.exists(path) for path in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH]):
        raise FileNotFoundError("❌ No se encontraron los archivos del modelo. Entrena el modelo primero.")

def load_model():
    check_model_files()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_classes = np.load(ENCODER_PATH, allow_pickle=True)
    return model, vectorizer, label_classes

def predict_emotion(text):
    model, vectorizer, label_classes = load_model()
    
    # Transformar el texto con el vectorizador
    text_vectorized = vectorizer.transform([text])

    # Predecir emoción
    prediction = model.predict(text_vectorized)
    emotion = label_classes[prediction[0]]

    return emotion

if __name__ == "__main__":
    while True:
        user_text = input("📝 Ingrese un texto para analizar (o 'exit' para salir): ")
        if user_text.lower() == "exit":
            break
        emotion = predict_emotion(user_text)
        print(f"😃 Emoción detectada: {emotion}")
'''🐢SA'''