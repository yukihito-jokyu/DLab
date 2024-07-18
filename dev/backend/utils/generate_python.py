def make_conv_layer(layer, in_channels):
    return f'            nn.Conv2d({in_channels}, {layer["out_channel"]}, {layer["kernel_size"]}, stride={layer["strid"]}, padding={layer["padding"]}),\n'

def make_pool_layer(pool_type, kernel_size, stride, padding):
    return f'            nn.{pool_type}({kernel_size}, stride={stride}, padding={padding}),\n'

def make_dropout_layer(dropout_p):
    return f'            nn.Dropout(p={dropout_p}),\n'

def make_batchnorm2d_layer(num_features):
    return f'            nn.BatchNorm2d({num_features}),\n'

def make_batchnorm1d_layer(num_features):
    return f'            nn.BatchNorm1d({num_features}),\n'

def make_linear(input_size, output_size):
    return f'            nn.Linear({input_size}, {output_size}),\n'

def make_activ(activ_name):
    return f'            nn.{activ_name}(),\n'

def make_flatten_bif(flutten_bif):
    return f'            nn.{flutten_bif}(1),\n'

def make_python_code(data):
    py_1 = '''
import torch.nn as nn
'''
    py_2 = '''
class Simple_NN(nn.Module):
    def __init__(self):
        super(Simple_NN, self).__init__()
'''
    py_3 = '''
        self.seq = nn.Sequential(
'''
    py_4 = '''
            )
'''
    py_5 = '''
    def forward(self, x):
        return self.seq(x)
'''
    structure = data.get('structure')
    input_layer = structure.get('InputLayer')
    conv_layers = structure.get('ConvLayer')
    middle_layers = structure.get('MiddleLayer')
    output_size = structure.get('OutputLayer')
    flatten_way = structure.get('FlattenWay')['way']
    save_dir = f'./user/{data.get("user_id")}/{data.get("project_name")}/{data.get("model_id")}'

    in_channels = input_layer['shape'][2]
    for layer in conv_layers:
        layer_type = layer.get('layer_type')
        if layer_type == 'Conv2d':
            py_3 += make_conv_layer(layer, in_channels)
            py_3 += make_activ(layer.get('activ_func'))
            in_channels = layer.get('out_channel')
        elif layer_type == 'MaxPool2d':
            py_3 += make_pool_layer(layer.get('layer_type'), layer.get('kernel_size'), layer.get('strid'), layer.get('padding'))
        elif layer_type == 'Dropout':
            py_3 += make_dropout_layer(layer.get('dropout_p'))
        elif layer_type == 'BatchNorm':
            py_3 += make_batchnorm2d_layer(in_channels)

    if flatten_way == 'GAP':
        py_3 += '            nn.AdaptiveAvgPool2d(1),\n'
    elif flatten_way == 'GMP':
        py_3 += '            nn.AdaptiveMaxPool2d(1),\n'

    py_3 += '            nn.Flatten(),\n'

    input_size = data.get('flattenshape')[0]

    for layer in middle_layers:
        layer_type = layer.get('layer_type')
        if layer_type == 'Neuron':
            py_3 += make_linear(input_size, layer.get('input_size'))
            py_3 += make_activ(layer.get('activ_func'))
            input_size = layer.get('input_size')
        if layer_type == 'Dropout':
            py_3 += make_dropout_layer(layer.get('dropout_p'))
        if layer_type == 'BatchNorm':
            py_3 += make_batchnorm1d_layer(input_size)

    py_3 += make_linear(input_size, output_size)

    python_code = py_1 + py_2 + py_3 + py_4 + py_5
    file_name = 'model_config.py'

    try:
        with open(f"{save_dir}/{file_name}", 'w') as file:
            file.write(python_code)
        return {"message": "successfully"}
    except Exception as e:
        return {"message": str(e)}
