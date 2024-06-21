# Flask
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# gym
from gyms.CartPole.train import cartpole
from gyms.CartPole.try_game import trycartpole
from gyms.FlappyBird.train import flappybird

# image
from Image.utils.util import handle_unzip, load_image, encode_image

# utils
from utils.resize_image import get_resize_image
from utils.generate_python import make_python_code

# test
from utils.test import test_env
from pygame_test.test import pygame_window

# ライブラリ
import time

# 非同期処理に使用するライブラリの指定
# `threading`, `eventlet`, `gevent`から選択可能
async_mode = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

# スレッドを格納するためのグローバル変数
thread = None

# test
@socketio.on('test_socket')
def test_socket(data):
    global thread
    id = data.get('id')
    print(id)
    print(thread)
    # if thread is None:
    print(thread)
    thread = socketio.start_background_task(target=test_(id))
    print(thread)
    emit('socket_test'+id, {'data': 0})

def test_(id):
    for i in range(10):
        socketio.sleep(1)
        emit('socket_test'+id, {'data': i+1})

@app.route('/test/pygame')
def test_pygame():
    pygame_window()
    return jsonify({'message': 'successfully'})

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

@app.route('/Reinforcement/Cartpole/download_pth')
def cortpole_download_pth():
    print('ダウンロード開始')
    pth_file_path = './weights/best_CartPole.pth'
    return send_file(pth_file_path, as_attachment=True)

@app.route('/Reinforcement/Cartpole/download_py')
def cortpole_download_py():
    print('ダウンロード開始')
    python_file_path = './python/model_config.py'
    return send_file(python_file_path, as_attachment=True)

# pth,pyのアップロード
@app.route('/CartPole/pth/upload', methods=['POST'])
def upload_cartpole_pth():
    file = request.files['file']
    file.save('./save/best_model.pth')
    return {'pth_success': True}

@app.route('/CartPole/py/upload', methods=['POST'])
def upload_cartpole_py():
    file = request.files['file']
    file.save('./save/model_config.py')
    return {'py_success': True}

@app.route('/CartPole/make/config', methods=['POST'])
def cartpole_make_config():
    data = request.get_json()
    print(data)
    make_python_code(data)
    return {'message': True}

@socketio.on('start_process')
def socket_process(data):
    print('開始')
    for i in range(10):
        time.sleep(0.1)
        emit('update', {'progress': i, 'data': f'Received data: {i}'})
    emit('complete', {'message': 'Processing complete!'})
    print('終わり')

@socketio.on('CartPole')
def train_CartPole(datas):
    # make_python_code(data)
    global thread
    data = datas.get('CartPole')
    id = datas.get('id')
    structures = data.get('structures')
    other_structure = data.get('other_structure')
    train_info = data.get('train_info')
    cartpole(structures, other_structure, train_info, id, socketio)
    # thread = socketio.start_background_task(cartpole(structures, other_structure, train_info, id))



@socketio.on('CartPoleTry')
def try_CartPole(data):
    trycartpole()

@socketio.on('FlappyBird')
def train_FlappyBird(datas):
    data = datas.get('alldata')
    id = datas.get('id')
    print(data)
    structures = data.get('Structure')
    other_structure = data.get('other_structure')
    train_info = data.get('train_info')
    flappybird(structures, other_structure, train_info, id)


@socketio.on('InputImage')
def resize_Flappy_image(data):
    get_resize_image(data)


@socketio.on('test_CartPole')
def test(data):
    test_env()
    emit('end_CartPole', {'message': 'Processing complete!'})

# zipファイルの取得
@app.route('/Image/UploadZip/train', methods=['POST'])
def UploadTrainZip():
    file = request.files['file']
    file.save('./Image/ZipFile/train.zip')
    return {'message': True}

@app.route('/Image/UploadZip/test', methods=['POST'])
def UploadTestZip():
    file = request.files['file']
    file.save('./Image/ZipFile/test.zip')
    return {'message': True}

@app.route('/Image/UnZip', methods=['POST'])
def UnZip():
    train_path = './Image/ZipFile/train.zip'
    test_path = './Image/ZipFile/test.zip'
    save_train_path = './Image/ImageFile'
    save_test_path = './Image/ImageFile'
    handle_unzip(train_path, save_train_path)
    handle_unzip(test_path, save_test_path)
    return {'message': True}

@app.route('/Image/LoadImage', methods=['POST'])
def Load_Image():
    train_path = './Image/ImageFile/train'
    test_path = './Image/ImageFile/test'
    train_images = load_image(train_path, 4)
    test_images = load_image(test_path, 2)
    encode_train_image = encode_image(train_images)
    encode_test_image = encode_image(test_images)
    res = {
        'train_image_list': encode_train_image,
        'test_image_list': encode_test_image
    }
    return res



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
