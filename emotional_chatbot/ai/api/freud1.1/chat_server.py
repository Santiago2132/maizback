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
import pymysql
from hashlib import scrypt
# from connect_bd import Database
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv #pip install python-dotenv
import secrets  # Para generar un salt aleatorio
import base64  # Para codificar el hash en base64
load_dotenv() # Carga las variables de entorno del .env
from google.oauth2 import id_token
from google.auth.transport import requests
import jwt  # Para generar tokens propios
import datetime

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
        'label_encoder': np.load("ai/models/label_encoder_classes.npy", allow_pickle=True) #CORREGIR POR LAS CONSULTAS DE BD
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
    stored_hash_bytes = base64.b64decode(stored_hash.encode('utf-8'))
    salt = stored_hash_bytes[:16]  # Extrae el salt de los primeros 16 bytes
    password_bytes = password_plain.encode('utf-8')
    new_hash = scrypt(password_bytes, salt=salt, n=16384, r=8, p=1, dklen=64)
    return new_hash == stored_hash_bytes[16:]  # Compara con la parte del hash

# -------------------------------- API LOGIN -----------------------------------------------

# Configuración de tu clave secreta para firmar tokens propios
SECRET_KEY = "TU_CLAVE_SECRETA_AQUI"
GOOGLE_CLIENT_ID = "TU_CLIENT_ID_AQUI"

# Ruta para insertar un nuevo usuario con app
@app.route('/registro/usuarios/app', methods=['POST'])
def insert_user():
    data = request.json
    nombre = data.get('nombre')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    if not nombre or not email or not password or not confirm_password:
        return jsonify({"error": "Todos los campos son obligatorios."}), 400

    if password != confirm_password:
        return jsonify({"error": "Las contraseñas no coinciden."}), 400

    # Encriptar la contraseña antes de almacenarla
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    connection = connect()
    try:
        cursor = connection.cursor()

        # Verificar si el usuario ya está registrado
        cursor.execute("SELECT id FROM usuarios WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({"error": "El email ya está registrado."}), 409

        # Insertar el nuevo usuario
        query = "INSERT INTO usuarios (nombre, email, contrasena) VALUES (%s, %s, %s)"
        cursor.execute(query, (nombre, email, hashed_password))
        connection.commit()

        return jsonify({"message": "Registro exitoso."}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

# Ruta para insertar un nuevo usuario con google
@app.route('/registro/usuarios/google', methods=['POST'])
def insert_user_google():
    if not request.is_json:
        return jsonify({"error": "El Content-Type debe ser 'application/json'"}), 415

    data = request.get_json()
    id_token_google = data.get('id_token')

    if not id_token_google:
        return jsonify({"error": "Se requiere un 'id_token' válido de Google."}), 400

    try:
        # Verificar el ID Token con Google
        google_user = id_token.verify_oauth2_token(id_token_google, requests.Request(), GOOGLE_CLIENT_ID)
        google_id = google_user.get('sub')  # ID único de Google
        email = google_user.get('email')
        name = google_user.get('name')

        if not google_id or not email or not name:
            return jsonify({"error": "No se pudo extraer información del ID Token."}), 400

        connection = connect()
        if connection is None:
            return jsonify({"error": "No se pudo conectar a la base de datos."}), 500

        cursor = connection.cursor(dictionary=True)

        # Verificar si el usuario ya existe
        cursor.execute("SELECT id FROM usuarios WHERE google_id = %s OR email = %s", (google_id, email))
        usuario = cursor.fetchone()

        if usuario:
            return jsonify({"message": "Inicio de sesión exitoso.", "user_id": usuario["id"]}), 200

        # Si no existe, registrarlo sin contraseña
        insert_user_query = "INSERT INTO usuarios (google_id, email, nombre) VALUES (%s, %s, %s)"
        cursor.execute(insert_user_query, (google_id, email, name))
        connection.commit()
        user_id = cursor.lastrowid

        return jsonify({"message": "Usuario registrado exitosamente con Google.", "user_id": user_id}), 201

    except ValueError:
        return jsonify({"error": "El ID Token de Google no es válido."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()

# Ruta para verificar si un usuario ya está registrado
@app.route('/verificar/usuarios', methods=['POST'])
def check_user():
    if not request.is_json:
        return jsonify({"error": "El Content-Type debe ser 'application/json'"}), 415

    data = request.get_json()
    email = data.get('email')
    google_id = data.get('google_id')

    if not email and not google_id:
        return jsonify({"error": "Se requiere al menos un campo: email o google_id."}), 400

    connection = connect()
    if connection is None:
        return jsonify({"error": "No se pudo conectar a la base de datos."}), 500

    try:
        cursor = connection.cursor()
        query = "SELECT id, email, google_id FROM usuarios WHERE "
        conditions = []
        values = []

        if email:
            conditions.append("email = %s")
            values.append(email)
        if google_id:
            conditions.append("google_id = %s")
            values.append(google_id)

        query += " AND ".join(conditions)  # Se usa AND para evitar falsos positivos
        cursor.execute(query, values)
        usuario = cursor.fetchone()

        if usuario:
            user_data = {"user_id": usuario[0], "email": usuario[1], "google_id": usuario[2]}
            return jsonify({"message": "El usuario ya está registrado.", "usuario": user_data}), 200
        else:
            return jsonify({"message": "El usuario no está registrado."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if cursor:
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

    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT id, email, contrasena FROM usuarios WHERE email = %s"
        cursor.execute(query, (email,))
        usuario = cursor.fetchone()

        if not usuario:
            return jsonify({"error": "El usuario no está registrado."}), 404

        # Verificar la contraseña
        stored_password_hash = usuario['contrasena']
        if check_password(password, stored_password_hash):
            return jsonify({"message": "Inicio de sesión exitoso."}), 200
        else:
            return jsonify({"error": "Contraseña incorrecta."}), 401

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

# Ruta para iniciar sesión con google
@app.route('/login/google', methods=['POST'])
def google_login():
    try:
        data = request.json
        token = data.get('token')

        if not token:
            return jsonify({"error": "Token de Google requerido"}), 400

        # Verificar el token con Google
        try:
            idinfo = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        except ValueError:
            return jsonify({"error": "Token de Google inválido"}), 401

        # Extraer información del usuario
        google_id = idinfo['sub']  # ID único de Google
        email = idinfo['email']
        name = idinfo.get('name', '')

        connection = connect()
        cursor = connection.cursor(dictionary=True)

        # Verificar si el usuario ya está registrado
        cursor.execute("SELECT id FROM usuarios WHERE google_id = %s", (google_id,))
        user = cursor.fetchone()

        if user:
            user_id = user['id']
        else:
            # Si el usuario no existe, registrarlo en la base de datos
            cursor.execute("INSERT INTO usuarios (google_id, email, nombre) VALUES (%s, %s, %s)", (google_id, email, name))
            connection.commit()
            user_id = cursor.lastrowid

        # Generar un token JWT propio para el usuario
        token_payload = {
            "user_id": user_id,
            "email": email,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)  # Expira en 24 horas
        }
        user_token = jwt.encode(token_payload, SECRET_KEY, algorithm="HS256")

        return jsonify({"message": "Inicio de sesión exitoso", "token": user_token, "user_id": user_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

@app.route('/usuarios/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    if not request.is_json:
        return jsonify({"error": "El Content-Type debe ser 'application/json'"}), 415

    data = request.get_json()
    email = data.get('email')
    nombre = data.get('nombre') 
    contrasena_plain = data.get('contrasena') 

    if not any([email, nombre, contrasena_plain]):
        return jsonify({"error": "No se proporcionaron campos para actualizar."}), 400

    connection = connect()
    if connection is None:
        return jsonify({"error": "No se pudo conectar a la base de datos."}), 500

    try:
        cursor = connection.cursor()
        update_fields = []
        values = []

        if email:
            update_fields.append("email = %s")
            values.append(email)
        if nombre:
            update_fields.append("nombre = %s")
            values.append(nombre)
        if contrasena_plain:
            contrasena_hash = generar_hash_contraseña(contrasena_plain)
            update_fields.append("contrasena = %s")
            values.append(contrasena_hash)

        query = f"""
        UPDATE usuarios 
        SET {', '.join(update_fields)} 
        WHERE id = %s
        """
        values.append(user_id)

        cursor.execute(query, values)
        if cursor.rowcount == 0:
            return jsonify({"error": "No se encontró el usuario o no hubo cambios."}), 404

        connection.commit()
        return jsonify({"message": f"Usuario con ID {user_id} actualizado correctamente."}), 200

    except Exception as e:
        connection.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

# -------------------------------- API INICIO -----------------------------------------------


# ------------------------------ API CALENDARIO ---------------------------------------------


# --------------------------------- API CHAT ------------------------------------------------


# ------------------------------- API TU DIA ------------------------------------------------


# ------------------------------- API PERFIL ------------------------------------------------

if __name__ == '__main__':
    monitor_thread = threading.Thread(target=monitor_system, daemon=True)
    monitor_thread.start()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 4000)), debug=False) 