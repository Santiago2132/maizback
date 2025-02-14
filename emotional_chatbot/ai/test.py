import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar el modelo y los recursos necesarios
model = joblib.load("ai/emotion_model.pkl")
vectorizer = joblib.load("ai/vectorizer.pkl")
label_classes = np.load("ai/label_encoder_classes.npy", allow_pickle=True)

def predict_emotion(text):
    """ Predice la emoción de un texto y devuelve la etiqueta correspondiente. """
    X_input = vectorizer.transform([text])
    emotion_index = model.predict(X_input)[0]
    return label_classes[emotion_index]

if __name__ == "__main__":
    while True:
        user_text = input("Ingrese un texto para probar (o escriba 'salir' para terminar): ")
        if user_text.lower() in ["salir", "exit", "quit"]:
            break
        emotion = predict_emotion(user_text)
        print(f"🔹 Emoción detectada: {emotion}")
