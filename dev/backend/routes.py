# from flask import render_template
from firebase_config import initialize_firebase, get_firestore_db

def setup_routes_base(app):
    @app.route('/')
    def index():
        return {'message': True}


def setup_db(app):
    initialize_firebase()
    db = get_firestore_db()

    @app.route('/get_db', methods=['GET'])
    def get_data_from_firebase():
        docs = db.collection('user').get()
        for doc in docs:
            print(doc.to_dict())
        return {'message': True}