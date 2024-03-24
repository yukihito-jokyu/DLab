import zipfile
import os
import cv2
import base64

# zipファイル解凍
def handle_unzip(zip_path, save_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)

# 画像の読み込み
def load_image(folder_path, number):
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    return files[:number]

def encode_image(paths):
    encode_image_list = []
    for path in paths:
        image = cv2.imread(path)
        resized_image = cv2.resize(image, (100, 100))
        _, buffer = cv2.imencode('.png', resized_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        encode_image_list.append(encoded_image)
    return encode_image_list