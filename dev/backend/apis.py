from flask import Flask, render_template, request, jsonify
import numpy as np
# from flask_cors import CORS

# utils
from utils.req_dataset import dataset_imager, get_images

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
    
    @app.route('/api/pre_data', methods=['POST'])
    def pre_data():
        data = request.get_json()
        print(data)
        images, pre_images, label_list = get_images(data)
        sent_data = {
            'images': images,
            'pre_images': pre_images,
            'label_list': label_list
        }
        return jsonify(sent_data)
