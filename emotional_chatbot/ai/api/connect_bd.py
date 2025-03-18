import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv #pip install python-dotenv
load_dotenv() # Carga las variables de entorno del .env

class Database:
    def _init_(self):
        self.host = os.getenv("DB_HOST")
        self.port = int(os.getenv("DB_PORT")) 
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.database = os.getenv("DB_NAME")
        ssl_disabled=True  # Desactiva SSL temporalmente
        self.connection = None
        self.cursor = None

    def connect(self):#Conexión a la base de datos
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                # ssl_ca=self.ssl_cert
            )
            if self.connection.is_connected():
                print("Conexión exitosa a la base de datos en DigitalOcean")
                self.cursor = self.connection.cursor()
        except Error as e:
            print(f"Error al conectar a MySQL: {e}")

    def disconnect(self): #Cierra la Conexión a la base de datos
        if self.connection and self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            print("Conexión cerrada")

    def execute_query(self, query, params=None): #Ejecuta instrucciones Select y devuelve
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except Error as e:
            print(f"Error ejecutando consulta: {e}")
            return None

    def execute_modification(self, query, values): #Maneja: insert, update y delete
        try:
            self.cursor.execute(query, values)
            self.connection.commit()
            print("Operación completada")
        except Error as e:
            print(f"Error modificando datos: {e}")

if _name_ == "_main_":
    db = Database() #instancia
    db.connect() #llama la instancia y la conecta con la bd
    db.execute_query("SELECT DATABASE();")
    db.disconnect()#desconecta la bd