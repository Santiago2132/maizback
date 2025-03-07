import mysql.connector
from mysql.connector import Error

class Database:
    def __init__(self, host="127.0.0.1", port=3306, user="root", password="1234", database="maiz"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

    def connect(self): #Conexión a la base de datos
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
        except Error as e:
            print(f"Error connecting to MySQL: {e}")

    def disconnect(self): #Cierra la Conexión a la base de datos
        if self.connection and self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            print("Connection closed")

    def execute_query(self, query, params=None): #Ejecuta instrucciones Select y devuelve
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except Error as e:
            print(f"Error executing query: {e}")
            return None

    def execute_modification(self, query, values): #Maneja: insert, update y delete.
        try:
            self.cursor.execute(query, values)
            self.connection.commit()
            print("Operation completed successfully")
        except Error as e:
            print(f"Error modifying data: {e}")

if __name__ == "__main__":
    db = Database()
    db.connect()

    db.execute_query("SELECT DATABASE();")

    db.disconnect()


