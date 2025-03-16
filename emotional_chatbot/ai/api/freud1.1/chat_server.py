import os
import psutil
import threading
import time
import socket
from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
import json
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import fuzz
from rich.console import Console
from rich.table import Table

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Inicializar NLP
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("spanish"))

def get_local_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

# Agregar IPs locales por defecto
local_ip = get_local_ip()
active_ips = {"127.0.0.1", local_ip}
console = Console()

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
                response_map.setdefault(tag, []).extend(intent['responses'])
                patterns_by_intent.setdefault(tag, []).extend(intent['patterns'])
    return response_map, patterns_by_intent

def load_offensive_words():
    if os.path.exists("ai/data/dictionary_word_dataset.csv"):
        df = pd.read_csv("ai/data/dictionary_word_dataset.csv", header=None, encoding="utf-8")
        return {lemmatizer.lemmatize(word.lower()) for word in df[0].astype(str).tolist()}
    return set()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s]', '', text.lower().strip())
    tokens = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text) if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def monitor_system():
    process = psutil.Process(os.getpid())
    while True:
        table = Table(title="Monitoreo del Servidor")
        table.add_column("Parámetro", justify="left", style="cyan")
        table.add_column("Valor", justify="right", style="magenta")
        
        ram_usage = process.memory_info().rss / (1024 * 1024)  # Convertir a MB
        cpu_usage = process.cpu_percent(interval=1)
        
        table.add_row("RAM Usada", f"{ram_usage:.2f} MB")
        table.add_row("CPU Usada", f"{cpu_usage:.2f}%")
        table.add_row("IPs Activas", ", ".join(active_ips) if active_ips else "Ninguna")
        
        console.clear()
        console.print(table)
        time.sleep(2)

# Configuración del servidor Flask
app = Flask(__name__)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    client_ip = request.remote_addr
    active_ips.add(client_ip)
    
    if not user_message:
        return jsonify({"error": "No se proporcionó un mensaje"}), 400
    
    bot_response = "Hola, esta es una respuesta de prueba"  # Simulación de respuesta
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    monitor_thread.start()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 4000)), debug=False)