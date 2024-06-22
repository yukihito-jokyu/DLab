from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    socketio.init_app(app)

    with app.app_context():
        # ブループリントやルートを登録
        from .routes import main as main_blueprint
        app.register_blueprint(main_blueprint)

    return app
