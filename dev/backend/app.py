# Flaskアプリのコード (backend/app/app.pyなど)
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    data = ['Item 1', 'Item 2', 'Item 3']
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
