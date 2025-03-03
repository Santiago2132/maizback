import os
import json
import joblib
import numpy as np
import tensorflow as tf
import nltk
import re  # Importar 're' para expresiones regulares
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder  # Importar LabelEncoder

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar lematizador y stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Cargar modelos y recursos
mlp_model = tf.keras.models.load_model("ai/models/emotion_mlp_model.keras")
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy')  # Compilar para evitar advertencias

rf_model = joblib.load("ai/models/emotion_rf_model.pkl")
vectorizer = joblib.load("ai/models/vectorizer.pkl")
label_classes = np.load("ai/models/label_encoder_classes.npy", allow_pickle=True)

# Reconstruir LabelEncoder
le = LabelEncoder()
le.classes_ = label_classes

# Cargar respuestas predefinidas desde múltiples archivos JSON
response_map = {}
intents_files = ["ai/data/intents.json", "ai/data/extra_intents.json"]

for file in intents_files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            intents = json.load(f)
        for intent in intents['intents']:
            # Mapear cada etiqueta a sus posibles respuestas
            response_map[intent['tag']] = intent['responses']

def text_preprocessing(text):
    """Preprocesamiento consistente con el entrenamiento"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Limpieza de texto utilizando expresiones regulares
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower().strip()
    
    # Tokenización y lematización
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    
    return ' '.join(tokens)

def predict_emotion(text):
    """Predicción con preprocesamiento consistente y alineación de probabilidades"""
    preprocessed_text = text_preprocessing(text)
    X_input = vectorizer.transform([preprocessed_text])
    
    # Obtener predicciones
    rf_pred = rf_model.predict(X_input)[0]
    mlp_probs = mlp_model.predict(X_input.toarray())[0]  # Probabilidades del MLP
    mlp_pred = np.argmax(mlp_probs)

    # Obtener probabilidades del Random Forest
    rf_probs_raw = rf_model.predict_proba(X_input)[0]
    
    # Crear un array de ceros del tamaño total de clases
    all_classes = label_classes
    rf_proba = np.zeros(len(all_classes))
    
    # Mapear las probabilidades del Random Forest a las posiciones correctas
    rf_class_labels = le.inverse_transform(rf_model.classes_)  # Obtener etiquetas de clase reales

    for idx_rf, class_label in enumerate(rf_class_labels):
        idx_all = np.where(all_classes == class_label)[0][0]
        rf_proba[idx_all] = rf_probs_raw[idx_rf]
    
    # Ahora rf_proba y mlp_probs están alineados
    combined_proba = (rf_proba + mlp_probs) / 2
    predicted_class = all_classes[np.argmax(combined_proba)]
    
    return predicted_class

def generate_response(user_input):
    """Genera una respuesta basada en la emoción detectada."""
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
