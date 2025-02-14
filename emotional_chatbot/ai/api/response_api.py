import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

# Inicializar Flask
app = Flask(__name__)

# Cargar modelos y recursos
try:
    model = joblib.load("ai/models/emotion_model.pkl")
    vectorizer = joblib.load("ai/models/vectorizer.pkl")
    label_classes = np.load("ai/models/label_encoder_classes.npy", allow_pickle=True)
    
    with open("ai/data/intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)
    
    response_map = {intent['tag']: intent['responses'] for intent in intents['intents']}
except Exception as e:
    print(f"Error cargando modelos o datos: {e}")
    model, vectorizer, label_classes, response_map = None, None, None, {}


def predict_emotion(text):
    """ Predice la emoción de un texto y devuelve la etiqueta correspondiente. """
    X_input = vectorizer.transform([text])
    emotion_index = model.predict(X_input)[0]
    return label_classes[emotion_index]


def generate_response(user_input):
    """ Genera una respuesta basada en la emoción detectada. """
    emotion = predict_emotion(user_input)
    return np.random.choice(response_map.get(emotion, ["Lo siento, no entendí tu emoción. ¿Puedes decirlo de otra manera?"]))


@app.route('/chat', methods=['POST'])
def chat():
    if any(x is None for x in (model, vectorizer)) or label_classes is None:
        return jsonify({"error": "Los modelos no se cargaron correctamente"}), 500
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "No se proporcionó un mensaje"}), 400
        
        bot_response = generate_response(user_message)
        return jsonify({"response": bot_response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)