import cv2
import torch
import numpy as np
import base64
from flask_socketio import emit

def make_black_white_image(image, H, W):
    # image = np.transpose(image, (1, 0, 2))
    image = cv2.resize(image, (W, H))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    image = np.reshape(image, (H, W, 1))
    tensor_image = torch.from_numpy(image).float()
    tensor_image = tensor_image.permute(2, 0, 1)
    return tensor_image / 255.0

def make_color_image(image, H, W):
    image = cv2.resize(image, (H, W))
    tensor_image = torch.from_numpy(image).float()
    tensor_image = tensor_image.permute(2, 0, 1)
    return tensor_image / 255.0

def get_make_image(C):
    if C == 1:
        return make_black_white_image
    if C == 3:
        return make_color_image

def get_resize_image(data):
    H = int(data.get('H'))
    W = int(data.get('W'))
    C = int(data.get('C'))
    image = cv2.imread('./assets/images/FlappyBird.png')
    if C == 1:
        image = make_black_white_image(image, H, W) * 255
        image = image.numpy().transpose(1, 2, 0)
    if C == 3:
        image = make_color_image(image, H, W) * 255
        image = image.numpy().transpose(1, 2, 0)
    _, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    # ソケット通信
    emit('get_resize_image', {'image_data': encoded_image})
