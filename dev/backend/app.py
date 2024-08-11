from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO

from routes import setup_routes_base, setup_db
from sockets import setup_sockets
from apis import setup_apis


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

setup_routes_base(app=app)
setup_db(app=app)
setup_apis(app=app)

setup_sockets(socketio=socketio)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
