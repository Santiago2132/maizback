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
from flask import Flask, request, jsonify
# from connect_bd import Database
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv #pip install python-dotenv
import secrets  # Para generar un salt aleatorio
import base64  # Para codificar el hash en base64
from hashlib import scrypt
load_dotenv() # Carga las variables de entorno del .env
import jwt # Para generar tokens propios
import datetime

# Descargamos recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración inicial
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("spanish"))
DEBUG = True  # Cambiar a False para desactivar mensajes de depuración

# Función para cargar modelos
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

# Función para cargar datos de intenciones
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
    
    if DEBUG:
        print("\nIntenciones cargadas:")
        for tag, patterns in patterns_by_intent.items():
            print(f"Tag: {tag} | Patrones: {patterns[:2]}... ({len(patterns)} patrones)")
    
    return response_map, patterns_by_intent

# Función para cargar palabras ofensivas
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
        if DEBUG:
            print(f"Comparando: '{input_text}' vs '{pattern}' → {similarity}%")
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
    
    if confidence < 0.6:  # Umbral de confianza
        return "unknown"
    
    return models['label_encoder'].classes_[predicted_class_idx]

# Detección de palabras ofensivas
def detect_offensive_words(text, offensive_words):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return any(word in offensive_words for word in tokens)

# Generación de respuesta
def generate_response(user_input, models, response_map, patterns_by_intent, offensive_words):
    # Detección de palabras ofensivas
    if detect_offensive_words(user_input, offensive_words):
        return "Lo siento, pero mantengamos nuestra conversación respetuosa."
    
    # Búsqueda por coincidencia de patrones
    for tag, patterns in patterns_by_intent.items():
        if is_similar_to_intent(user_input, patterns, 65):
            if DEBUG:
                print(f"Coincidencia por patrón con tag: {tag}")
            return np.random.choice(response_map[tag])
    
    # Predicción por modelo ML
    emotion = predict_emotion(user_input, models)
    if DEBUG:
        print(f"Predicción del modelo: {emotion}")
    
    if emotion != "unknown" and emotion in response_map:
        return np.random.choice(response_map[emotion])
    
    return "No estoy seguro de qué decirte... ¿Puedes darme más detalles?"

# Inicialización del sistema
def initialize_system():
    models = load_models()
    offensive_words = load_offensive_words()
    response_map, patterns_by_intent = load_intents()
    return models, offensive_words, response_map, patterns_by_intent

# Configuración del servidor Flask
app = Flask(__name__)

# Cargar modelos y datos al iniciar el servidor
models, offensive_words, response_map, patterns_by_intent = initialize_system()

# --------------------- CONEXIÓN A BASE DE DATOS -------------------------
def connect():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 3306)),  # Puerto por defecto
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        if connection.is_connected():
            print("Conexión exitosa a la base de datos")
            return connection
    except Error as e:
        print(f"Error al conectar a MySQL: {e}")
        return None

# ----------------------- HASH DE CONTRASEÑAS -------------------------
def generate_hash_password(password_plain):
    salt = secrets.token_bytes(16)  # Genera un salt aleatorio
    password_bytes = password_plain.encode('utf-8')
    password_hash = scrypt(password_bytes, salt=salt, n=16384, r=8, p=1, dklen=64)
    return base64.b64encode(salt + password_hash).decode('utf-8')  # Guardamos salt + hash

def check_password(password_plain, stored_hash):
    missing_padding = len(stored_hash) % 4
    if missing_padding:
        stored_hash += '=' * (4 - missing_padding)
    stored_hash_bytes = base64.b64decode(stored_hash.encode('utf-8'))
    salt = stored_hash_bytes[:16]
    password_bytes = password_plain.encode('utf-8')
    new_hash = scrypt(password_bytes, salt=salt, n=16384, r=8, p=1, dklen=64)
    return new_hash == stored_hash_bytes[16:]  # Compara con la parte del hash


# -------------------------------- API LOGIN -----------------------------------------------

# SECRET_KEY = "tu_clave_secreta_super_segura"
SECRET_KEY = "holi"

# Ruta para insertar un nuevo usuario con app
@app.route('/registro/app', methods=['POST'])
def insert_user():
    if not request.is_json:
        return jsonify({"error": "El Content-Type debe ser 'application/json'"}), 415

    data = request.get_json()

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    confirm_password = data.get("confirm_password")

    if not email or not name or not password or not confirm_password:
        return jsonify({"error": "Faltan datos obligatorios (email, name, password o confirm_password)."}), 400

    if password != confirm_password:
        return jsonify({"error": "Las contraseñas no coinciden."}), 400

    try:
        connection = connect()
        if connection is None:
            return jsonify({"error": "No se pudo conectar a la base de datos."}), 500

        cursor = connection.cursor(dictionary=True)

        # Verificar si el usuario ya existe
        cursor.execute("SELECT id, contrasena FROM usuarios WHERE email = %s", (email,))
        usuario = cursor.fetchone()

        if usuario:
            stored_password = usuario["contrasena"]
            if check_password(password, stored_password):
                return jsonify({"message": "El usuario ya está registrado", "user_id": usuario["id"]}), 200
            else:
                return jsonify({"error": "Contraseña incorrecta."}), 401

        # Crear nuevo usuario
        hashed_password = generate_hash_password(password)
        insert_user_query = "INSERT INTO usuarios (nombre, email, contrasena) VALUES (%s, %s, %s)"
        cursor.execute(insert_user_query, (name, email, hashed_password))
        connection.commit()
        user_id = cursor.lastrowid

        # Obtener ID de la insignia 'Bienvenido'
        cursor.execute("SELECT id FROM insignias WHERE nombre = %s", ("Bienvenido",))
        bienvenido = cursor.fetchone()

        if bienvenido:
            insert_insignia_query = """
                INSERT INTO insignias_usuario (id_usuario, id_insignia)
                VALUES (%s, %s)
            """
            cursor.execute(insert_insignia_query, (user_id, bienvenido["id"]))
            connection.commit()

        return jsonify({"message": "Usuario registrado exitosamente con app.", "user_id": user_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

# Ruta para insertar un nuevo usuario con google
@app.route('/registro/google', methods=['POST'])
def insert_user_google():
    if not request.is_json:
        return jsonify({"error": "El Content-Type debe ser 'application/json'"}), 415

    data = request.get_json()

    name = data.get("name")
    email = data.get("email")
    google_id = data.get("google_id")

    if not google_id or not email or not name:
        return jsonify({"error": "Faltan datos obligatorios (google_id, email o name)."}), 400

    try:
        connection = connect()
        if connection is None:
            return jsonify({"error": "No se pudo conectar a la base de datos."}), 500

        cursor = connection.cursor(dictionary=True)

        # Verificar si el usuario ya existe
        cursor.execute("SELECT id FROM usuarios WHERE google_id = %s OR email = %s", (google_id, email))
        usuario = cursor.fetchone()

        if usuario:
            return jsonify({"message": "El usuario ya está registrado.", "user_id": usuario["id"]}), 200

        # Si no existe, registrarlo
        default_password = secrets.token_hex(16)
        insert_user_query = "INSERT INTO usuarios (nombre, email, google_id, contrasena) VALUES (%s, %s, %s, %s)"
        cursor.execute(insert_user_query, (name, email, google_id, default_password))
        connection.commit()
        user_id = cursor.lastrowid

        # Asignar la insignia "Bienvenido"
        cursor.execute("SELECT id FROM insignias WHERE nombre = %s", ("Bienvenido",))
        bienvenido = cursor.fetchone()

        if bienvenido:
            insert_insignia_query = """
                INSERT INTO insignias_usuario (id_usuario, id_insignia)
                VALUES (%s, %s)
            """
            cursor.execute(insert_insignia_query, (user_id, bienvenido["id"]))
            connection.commit()

        return jsonify({"message": "Usuario registrado exitosamente con Google.", "user_id": user_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

# Ruta para iniciar sesión con app
@app.route('/login/app', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Se requieren tanto el email como la contraseña."}), 400

    connection = connect()
    if connection is None:
        return jsonify({"error": "No se pudo conectar a la base de datos."}), 500

    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT id, email, nombre, contrasena FROM usuarios WHERE email = %s"
        cursor.execute(query, (email,))
        usuario = cursor.fetchone()

        if not usuario:
            return jsonify({"error": "El usuario no está registrado."}), 404

        stored_password_hash = usuario.get('contrasena')
        if not stored_password_hash:
            return jsonify({"error": "El usuario no tiene una contraseña registrada."}), 500

        if not check_password(password, stored_password_hash):
            return jsonify({"error": "Contraseña incorrecta."}), 401

        token = jwt.encode(
            {
                "user_id": usuario["id"],
                "email": usuario["email"],
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=168)
            },
            SECRET_KEY,
            algorithm="HS256"
        )

        return jsonify({
            "message": "Inicio de sesión exitoso en API.",
            "user": {
                "id": usuario["id"],
                "name": usuario["nombre"]
            },
            "token": token
        }), 200

    except Exception as e:
        print(f"Error en login: {e}")
        return jsonify({"error": "Error interno en el servidor."}), 500

    finally:
        cursor.close()
        connection.close()

# -------------------------------- API INICIO -----------------------------------------------


# ------------------------------ API CALENDARIO ---------------------------------------------


# --------------------------------- API CHAT ------------------------------------------------

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "No se proporcionó un mensaje"}), 400
        
        bot_response = generate_response(
            user_input=user_message,
            models=models,
            response_map=response_map,
            patterns_by_intent=patterns_by_intent,
            offensive_words=offensive_words
        )
        
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------- API TU DIA ------------------------------------------------


# ------------------------------- API PERFIL ------------------------------------------------



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)