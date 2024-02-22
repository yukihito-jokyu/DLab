import torch.nn.functional as F

def get_activation(activation_name):
    if activation_name == 'ReLU':
        return F.relu
    if activation_name == 'Sigmoid':
        return F.sigmoid
    if activation_name == 'Tanh':
        return F.tanh
    if activation_name == 'Softmax':
        return F.softmax