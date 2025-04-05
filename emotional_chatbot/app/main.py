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
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración inicial
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("spanish"))
DEBUG = True

# Cargar modelos de clasificación
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

# Cargar datos de intenciones
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
    return response_map, patterns_by_intent

# Cargar palabras ofensivas
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
    if confidence < 0.6:
        return "unknown"
    return models['label_encoder'].classes_[predicted_class_idx]

# Generar párrafo con GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")
gen_model = GPT2LMHeadModel.from_pretrained("datificate/gpt2-small-spanish")
tokenizer.pad_token = tokenizer.eos_token  # Configurar pad_token

def generate_paragraph(prompt, max_new_tokens=50):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        max_length = min(512, len(input_ids[0]) + max_new_tokens)
        outputs = gen_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        if DEBUG:
            print(f"Error en generate_paragraph: {str(e)}")
        return None

# Historial de conversación
conversation_history = []

def add_to_history(role, message):
    if len(message) > 100:
        message = message[:100] + "..."
    conversation_history.append(f"{role}: {message}")
    if len(conversation_history) > 3:
        conversation_history.pop(0)

# Respuestas de respaldo por emoción
backup_responses = {
    "tristeza": "Lamento mucho que te sientas así. A veces, los días grises parecen eternos, pero siempre hay algo que puede ayudarte a ver la luz. ¿Te gustaría hablar de lo que te tiene así?",
    "felicidad": "¡Qué alegría saber que estás bien! Me encanta cuando compartes cosas positivas. ¿Qué ha hecho tu día tan especial?",
    "enojo": "Entiendo que estés molesto, y está bien sentirlo. ¿Hay algo específico que te esté sacando de quicio? Puedo escucharte.",
    "miedo": "Sé que el miedo puede ser abrumador. ¿Hay algo en particular que te esté preocupando? Estoy aquí para ayudarte a enfrentarlo.",
    "sorpresa": "¡Vaya, eso suena inesperado! Me tienes intrigado. ¿Qué pasó para que te sorprendieras tanto?",
    "neutral": "Gracias por contarme. Parece que estás en un momento tranquilo. ¿Hay algo más que quieras explorar juntos?",
    "unknown": "No estoy seguro de cómo te sientes ahora, pero quiero ayudarte. ¿Puedes darme más detalles sobre lo que pasa por tu mente?"
}

# Generación de respuesta
def generate_response(user_input, models, response_map, patterns_by_intent, offensive_words):
    if detect_offensive_words(user_input, offensive_words):
        return "Por favor, mantengamos el respeto en nuestra conversación."
    
    for tag, patterns in patterns_by_intent.items():
        if is_similar_to_intent(user_input, patterns, 65):
            return np.random.choice(response_map[tag])
    
    emotion = predict_emotion(user_input, models)
    add_to_history("Usuario", user_input)
    
    if conversation_history:
        history_summary = " ".join(conversation_history[-2:])
    else:
        history_summary = ""
    prompt = f"El usuario dice: '{user_input}'. Está {emotion}. Responde con empatía."
    if history_summary:
        prompt = f"Contexto: {history_summary}\n" + prompt
    
    response = generate_paragraph(prompt, max_new_tokens=50)
    
    if response:
        response = response.replace(prompt, "").strip()
        add_to_history("Bot", response)
        return response
    else:
        return backup_responses.get(emotion, "No sé qué decir exactamente, pero estoy aquí. ¿Puedes darme más contexto?")

# Detectar palabras ofensivas
def detect_offensive_words(text, offensive_words):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return any(word in offensive_words for word in tokens)

# Inicialización del sistema
def initialize_system():
    models = load_models()
    offensive_words = load_offensive_words()
    response_map, patterns_by_intent = load_intents()
    return models, offensive_words, response_map, patterns_by_intent

# Ejecución principal
if __name__ == "__main__":
    models, offensive_words, response_map, patterns_by_intent = initialize_system()
    
    print("\nSistema listo. Escribe 'exit' para terminar.\n")
    while True:
        try:
            user_input = input("Tú: ").strip()
            if user_input.lower() in ["exit", "quit", "salir"]:
                print("Bot: ¡Hasta pronto! Me encantó charlar contigo.")
                break
                
            response = generate_response(
                user_input=user_input,
                models=models,
                response_map=response_map,
                patterns_by_intent=patterns_by_intent,
                offensive_words=offensive_words
            )
            print(f"Bot: {response}\n")
            
        except Exception as e:
            print(f"Bot: Ups, algo salió mal. ¿Puedes intentarlo de nuevo?")
            if DEBUG:
                print(f"Error: {str(e)}")