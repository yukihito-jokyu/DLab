# backend/app/app.py
from flask import Flask, jsonify, request, stream_with_context
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# gym
from gyms.CartPole.train import cartpole

import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

@app.route('/', methods=['POST'])
def get_data():
    data = request.get_json()
    data['count'] += 1
    return jsonify(data)

@app.route('/train', methods=['POST'])
def get_train_data():
    data = request.get_json()
    print(data)
    return jsonify({'message': 'Data received successfully'})

@socketio.on('start_process')
def socket_process(data):
    print('開始')
    for i in range(10):
        time.sleep(0.1)
        emit('update', {'progress': i, 'data': f'Received data: {i}'})
    emit('complete', {'message': 'Processing complete!'})
    print('終わり')

@socketio.on('CartPole')
def train_CartPole(data):
    structures = data.get('structures')
    other_structure = data.get('other_structure')
    train_info = data.get('train_info')
    cartpole(structures, other_structure)
    pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
