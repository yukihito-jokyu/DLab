# Flask
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# thread
import threading
import concurrent.futures

# test
from pygame_test.test import pygame_window
from gyms.FlappyBird.train_test import flappytest

# 非同期処理に使用するライブラリの指定
# `threading`, `eventlet`, `gevent`から選択可能
async_mode = 'eventlet'
# async_mode = None


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

@app.route('/test/pygame')
def test_pygame():
  flappytest()
  
  return jsonify({'message': 'successfully'})


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=5050)