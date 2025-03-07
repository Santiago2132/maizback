import mysql.connector
from mysql.connector import Error

try:
    connection = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="1234",  # Reemplaza con la contraseña correcta
        database="maiz"  # Especifica la base de datos
    )

    if connection.is_connected():
        print("Conexión exitosa a MySQL")
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE();")
        record = cursor.fetchone()
        print(f"Conectado a la base de datos: {record[0]}")

except Error as e:
    print(f"Error al conectar a MySQL: {e}")

finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("Conexión cerrada")

