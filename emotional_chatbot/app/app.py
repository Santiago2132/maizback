import os
from flask import Flask
from chat_server import chat_bp  # Importamos el blueprint del chat
from database import db_bp      # Importamos el blueprint de la base de datos
from werkzeug.middleware.proxy_fix import ProxyFix  # Para entornos con proxies

def create_app():
    # Inicialización de la aplicación Flask
    app = Flask(__name__)
    
    # Configuraciones básicas
    app.config['DEBUG'] = True  # Cambiar a False en producción
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'clave_secreta_default')
    
    # Aplicar middleware para proxies (útil para balanceo de carga futuro)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
    
    # Registrar blueprints
    app.register_blueprint(chat_bp, url_prefix='/chat')  # Rutas de chat
    app.register_blueprint(db_bp, url_prefix='/db')      # Rutas de base de datos
    
    # Ruta de salud para monitoreo
    @app.route('/health', methods=['GET'])
    def health_check():
        return {"status": "healthy", "message": "Servidor en funcionamiento"}, 200
    
    # Manejo de errores globales
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Ruta no encontrada"}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Error interno del servidor"}, 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port, threaded=True, debug=app.config['DEBUG'])