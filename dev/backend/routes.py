# from flask import render_template
from flask import request
from firebase_config import initialize_firebase, get_firestore_db
from utils.generate_python import make_python_code
from utils.mkdir import create_user_directory, create_project_directory

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