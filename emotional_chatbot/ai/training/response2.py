import os
import json
import joblib
import numpy as np
import tensorflow as tf
import nltk
import re  # Importamos 're' para expresiones regulares
import pandas as pd  # Importamos pandas para manejar archivos CSV
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Descargamos recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializamos lematizador y stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Cargamos modelos y recursos
mlp_model = tf.keras.models.load_model("ai/models/emotion_mlp_model.keras")
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy')  # Compilamos para evitar advertencias

rf_model = joblib.load("ai/models/emotion_rf_model.pkl")
vectorizer = joblib.load("ai/models/vectorizer.pkl")
label_classes = np.load("ai/models/label_encoder_classes.npy", allow_pickle=True)

# Reconstruimos el LabelEncoder
le = LabelEncoder()
le.classes_ = label_classes

# Cargamos respuestas predefinidas de varios archivos JSON
response_map = {}
intents_files = ["ai/data/intents.json", "ai/data/extra_intents.json"]

for file in intents_files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            intents = json.load(f)
        for intent in intents['intents']:
            # Mapeamos cada etiqueta a sus posibles respuestas
            response_map[intent['tag']] = intent['responses']

# Función para cargar las palabras ofensivas desde un archivo CSV
def load_offensive_words(file_path):
    """Carga palabras ofensivas de un archivo CSV."""
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, header=None, encoding="utf-8")
            # Aseguramos que todas las palabras estén en minúsculas
            return df[0].astype(str).str.lower().tolist()
        except Exception as e:
            print(f"Error al cargar palabras ofensivas: {e}")
            return []
    else:
        print(f"Archivo de palabras ofensivas no encontrado en {file_path}")
        return []

# Función para detectar palabras ofensivas en un mensaje
def detect_offensive_words(text, offensive_words):
    """Detecta palabras ofensivas en el texto de entrada."""
    tokens = nltk.word_tokenize(text.lower())
    # Opcionalmente, lematizamos las palabras para una mejor coincidencia
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    offensive_found = [word for word in tokens if word in offensive_words]
    return offensive_found

# Cargamos las palabras ofensivas
offensive_words = load_offensive_words("ai/data/dictionary_word_dataset.csv")

def text_preprocessing(text):
    """Preprocesamiento coherente con el entrenamiento."""
    # Limpiamos el texto usando expresiones regulares
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
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
    """Realiza la predicción con preprocesamiento coherente y alineación de probabilidades."""
    preprocessed_text = text_preprocessing(text)
    X_input = vectorizer.transform([preprocessed_text])
    
    # Obtenemos predicciones
    rf_pred = rf_model.predict(X_input)[0]
    mlp_probs = mlp_model.predict(X_input.toarray())[0]  # Probabilidades del MLP
    
    # Obtenemos probabilidades del Random Forest
    rf_probs_raw = rf_model.predict_proba(X_input)[0]
    
    # Creamos un arreglo de ceros del tamaño de las clases
    all_classes = label_classes
    rf_proba = np.zeros(len(all_classes))
    
    # Mapeamos las probabilidades de Random Forest a las posiciones correctas
    rf_class_labels = le.inverse_transform(rf_model.classes_)
    for idx_rf, class_label in enumerate(rf_class_labels):
        idx_all = np.where(all_classes == class_label)[0][0]
        rf_proba[idx_all] = rf_probs_raw[idx_rf]
    
    # Ahora rf_proba y mlp_probs están alineados
    combined_proba = (rf_proba + mlp_probs) / 2
    predicted_class = all_classes[np.argmax(combined_proba)]
    
    return predicted_class

def generate_response(user_input):
    """Genera una respuesta basada en la emoción detectada o maneja contenido ofensivo."""
    # Primero, verificamos si hay palabras ofensivas
    offensive_found = detect_offensive_words(user_input, offensive_words)
    if offensive_found:
        # Podríamos registrar las palabras ofensivas aquí si es necesario
        return "Lo siento, pero mantengamos nuestra conversación respetuosa."
    
    # Continuamos con la predicción de emoción y generación de respuesta
    emotion = predict_emotion(user_input)
    
    if emotion in response_map:
        return np.random.choice(response_map[emotion])  # Elegimos una respuesta aleatoria de la lista
    
    return "Lo siento, no entendí bien. ¿Podrías reformular tu pregunta?"

if __name__ == "__main__":
    while True:
        user_text = input("Tú: ")
        if user_text.lower() in ["exit", "quit"]:
            print("Bot: ¡Adiós!")
            break
        response = generate_response(user_text)
        print(f"Bot: {response}")
