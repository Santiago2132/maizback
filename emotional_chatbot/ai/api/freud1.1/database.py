import mysql.connector # type: ignore
from mysql.connector import Error # type: ignore
from flask import Blueprint, jsonify

# Definimos el blueprint para las rutas de la base de datos
db_bp = Blueprint('db', __name__)

class Database:
    def __init__(self, host="127.0.0.1", port=3306, user="root", password="1234", database="maiz"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                print("Successfully connected to MySQL")
                self.cursor.execute("SELECT DATABASE();")
                print(f"Connected to the database: {self.cursor.fetchone()[0]}")
                return True
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            print("Connection closed")

    def execute_query(self, query, params=None):
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except Error as e:
            print(f"Error executing query: {e}")
            return None

    def execute_modification(self, query, values):
        try:
            self.cursor.execute(query, values)
            self.connection.commit()
            print("Operation completed successfully")
        except Error as e:
            print(f"Error modifying data: {e}")

# Instancia global de la base de datos
db = Database()

@db_bp.route('/connect', methods=['GET'])
def connect_db():
    if db.connect():
        return jsonify({"message": "Conexión establecida a la base de datos"}), 200
    return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

@db_bp.route('/disconnect', methods=['GET'])
def disconnect_db():
    db.disconnect()
    return jsonify({"message": "Conexión cerrada"}), 200

@db_bp.route('/test', methods=['GET'])
def test_db():
    db.connect()
    result = db.execute_query("SELECT DATABASE();")
    db.disconnect()
    if result:
        return jsonify({"database": result[0][0]}), 200
    return jsonify({"error": "Error al consultar la base de datos"}), 500