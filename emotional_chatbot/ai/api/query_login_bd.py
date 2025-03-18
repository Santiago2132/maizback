from flask import Flask, request, jsonify
import pymysql
from hashlib import scrypt
from connect_bd import Database #importa el archivo de la conexion

# Método para generar el hash de la contraseña
def generar_hash_contraseña(password_plain):
    password_bytes = password_plain.encode('utf-8')
    password_hash = scrypt(password_bytes, salt=b'some_salt', n=16384, r=8, p=1, dklen=64)
    return password_hash  

# Método para verificar la contraseña
def verificar_contraseña(password_plain, password_hash):
    password_bytes = password_plain.encode('utf-8')
    new_hash = scrypt(password_bytes, salt=b'some_salt', n=16384, r=8, p=1, dklen=64)
    return new_hash == password_hash


# Ruta para iniciar sesión
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Se requieren tanto el email como la contraseña."}), 400

    connection = Database.connect()

    try:
        cursor = connection.cursor()
        query = "SELECT USER_ID, EMAIL, PASSWORD FROM USERS WHERE EMAIL = %s"
        cursor.execute(query, (email,))
        usuario = cursor.fetchone()

        if not usuario:
            return jsonify({"error": "El usuario no está registrado."}), 404

        # Verificar la contraseña
        stored_password_hash = usuario['PASSWORD']
        if verificar_contraseña(password, stored_password_hash):
            return jsonify({"message": "Inicio de sesión exitoso."}), 200
        else:
            return jsonify({"error": "Contraseña incorrecta."}), 401

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()

# Ruta para verificar si un usuario ya está registrado
@app.route('/usuarios/verificar', methods=['POST'])
def verificar_usuario():
    data = request.json
    email = data.get('email')
    google_id = data.get('google_id')
    if not email and not google_id:
        return jsonify({"error": "Se requiere al menos un campo: email o google_id."}), 400
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        query = "SELECT USER_ID, EMAIL, GOOGLE_ID FROM USERS WHERE "
        conditions = []
        values = []
        if email:
            conditions.append("EMAIL = %s")
            values.append(email)
        if google_id:
            conditions.append("GOOGLE_ID = %s")
            values.append(google_id)
        query += " OR ".join(conditions)
        cursor.execute(query, values)
        usuario = cursor.fetchone()
        if usuario:
            return jsonify({"message": "El usuario ya está registrado.", "usuario": usuario}), 200
        else:
            return jsonify({"message": "El usuario no está registrado."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()

# Ruta para insertar un nuevo usuario
@app.route('/usuarios', methods=['POST'])
def insertar_usuario():
    data = request.json
    google_id = data.get('google_id')
    email = data.get('email')
    name = data.get('name')
    photo = data.get('photo')
    password_plain = data.get('password')

    if not password_plain:
        return jsonify({"error": "Se requiere una contraseña."}), 400

    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        password_hash = generar_hash_contraseña(password_plain)  
        insert_user_query = """
        INSERT INTO USERS (GOOGLE_ID, EMAIL, NAME, PHOTO, PASSWORD) 
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_user_query, (google_id, email, name, photo, password_hash))
        connection.commit()
        user_id = cursor.lastrowid
        return jsonify({"message": "Usuario insertado correctamente.", "user_id": user_id}), 201
    except Exception as e:
        if "Duplicate entry" in str(e):
            return jsonify({"error": "El email o google_id ya está registrado"}), 400
        else:
            return jsonify({"error": str(e)}), 500
    finally:
        connection.close()

# Ruta para insertar una entrada de ánimo
@app.route('/entradas-animo', methods=['POST'])
def insertar_entrada_animo():
    data = request.json
    user_id = data.get('user_id')
    mood = data.get('mood')
    date = data.get('date')
    note = data.get('note')
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        insert_mood_query = """
        INSERT INTO MOOD_ENTRIES (MOOD, DATE, NOTE, USER_ID) 
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_mood_query, (mood, date, note, user_id))
        connection.commit()
        return jsonify({"message": "Entrada de ánimo insertada correctamente."}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()

# Ruta para eliminar un usuario
@app.route('/usuarios/<int:user_id>', methods=['DELETE'])
def eliminar_usuario(user_id):
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        delete_mood_query = "DELETE FROM MOOD_ENTRIES WHERE USER_ID = %s"
        cursor.execute(delete_mood_query, (user_id,))
        delete_conversation_query = "DELETE FROM AI_CONVERSATION WHERE USER_ID = %s"
        cursor.execute(delete_conversation_query, (user_id,))
        delete_user_query = "DELETE FROM USERS WHERE USER_ID = %s"
        cursor.execute(delete_user_query, (user_id,))
        connection.commit()
        return jsonify({"message": f"Usuario con ID {user_id} y sus datos eliminados correctamente."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()

# Ruta para actualizar un usuario
@app.route('/usuarios/<int:user_id>', methods=['PUT'])
def actualizar_usuario(user_id):
    data = request.json
    email = data.get('email')
    name = data.get('name')
    photo = data.get('photo')
    password_plain = data.get('password')
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        if password_plain:
            password_hash = generar_hash_contraseña(password_plain)
            password_query = "PASSWORD = %s, "
            password_value = password_hash
        else:
            password_query = ""
            password_value = None
        update_fields = []
        values = []
        if email is not None:
            update_fields.append("EMAIL = %s")
            values.append(email)
        if name is not None:
            update_fields.append("NAME = %s")
            values.append(name)
        if photo is not None:
            update_fields.append("PHOTO = %s")
            values.append(photo)
        if password_value is not None:
            update_fields.append("PASSWORD = %s")
            values.append(password_value)
        if not update_fields:
            return jsonify({"error": "No se proporcionaron campos para actualizar."}), 400
        update_user_query = f"""
        UPDATE USERS 
        SET {', '.join(update_fields)} 
        WHERE USER_ID = %s
        """
        values.append(user_id)
        cursor.execute(update_user_query, values)
        connection.commit()
        return jsonify({"message": f"Usuario con ID {user_id} actualizado correctamente."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()

if __name__ == '__main__':
    app.run(debug=True)