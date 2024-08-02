# from flask import render_template
from flask import request, send_file, jsonify
from firebase_config import initialize_firebase, get_firestore_db
from utils.generate_python import make_python_code
from utils.mkdir import create_user_directory, create_project_directory, create_model_directory_from_dict
from utils.del_dir import delete_model_directories
from utils.download_model import download_model_directories

def setup_routes_base(app):
    @app.route('/')
    def index():
        return {'message': True}
    
    # userディレクトリ作成
    @app.route('/mkdir/user', methods=['POST'])
    def make_user_dir():
        data = request.get_json()
        print('message:', data)
        message = create_user_directory(data)
        return message
    
    # projectディレクトリ作成
    @app.route('/mkdir/project', methods=['POST'])
    def make_project_dir():
        data = request.get_json()
        print('data:', data)
        message = create_project_directory(data)
        return message
    
    # modelディレクトリ作成
    @app.route('/mkdir/model', methods=['POST'])
    def make_model_dir():
        data = request.get_json()
        print('data:', data)
        message = create_model_directory_from_dict(data)
        return message

    # modelディレクトリのダウンロード
    @app.route('/download_zip', methods=['POST'])
    def download_model_dir():
        data = request.json
        message = download_model_directories(data)
        return message    
    
    # modelディレクトリ削除
    @app.route('/del_dir/model', methods=['POST'])
    def delete_model_dir():
        data = request.get_json()
        print('data:', data)
        message = delete_model_directories(data)
        return message
    
    # 画像分類のモデルpyファイル作成
    @app.route('/ImageClassification/make/config', methods=['POST'])
    def imageclassification_make_config():
        data = request.get_json()
        print(data)
        message = make_python_code(data)
        print('作成！')
        print(message)
        return message
    
    # cartpoleのモデルpyファイル作成
    @app.route('/CartPole/make/config', methods=['POST'])
    def cartpole_make_config():
        data = request.get_json()
        print(data)
        message = make_python_code(data)
        return message


def setup_db(app):
    initialize_firebase()
    db = get_firestore_db()

    @app.route('/get_db', methods=['GET'])
    def get_data_from_firebase():
        docs = db.collection('user').get()
        for doc in docs:
            print(doc.to_dict())
        return {'message': True}