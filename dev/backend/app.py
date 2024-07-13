from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO

from train.train_cartpole import train_cartpole
from train.train_image_classification import train_model

from routes import setup_routes_base, setup_db
from sockets import setup_sockets


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

setup_routes_base(app=app)
setup_db(app=app)

setup_sockets(socketio=socketio)

@socketio.on('CartPole')
def train_CartPole(datas):
    global thread
    train_cartpole(datas, socketio)

@socketio.on('Train')
def train(datas):
    global thread
    train_model(datas, socketio)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
