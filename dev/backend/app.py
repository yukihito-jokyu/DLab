from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)