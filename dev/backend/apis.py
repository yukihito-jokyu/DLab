from flask import Flask, render_template, request, jsonify
import numpy as np
# from flask_cors import CORS

# utils
from utils.req_dataset import dataset_imager

def setup_apis(app):
    @app.route('/api/test', methods=['GET'])
    def status():
        return {'status': 'ok'}

    @app.route('/api/req_dataset', methods=['POST'])
    def req_dataset():
        # request
        req = request.json
        dataset_name = req['dataset']
        n = req['n']
        label = req['label']

        # load
        di = dataset_imager(dataset_name=dataset_name, n=n, label=label)
        if di.x is np.array([None]):
            return {'status': 'error'}
        else:
            images = di.dispend_image()
            return jsonify({'images': images})