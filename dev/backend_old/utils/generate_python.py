def make_linear(input_size, output_size):
    return f'            nn.Linear({input_size}, {output_size}),\n'

def make_activ(activ_name):
    return f'            nn.{activ_name}(),\n'

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
    structures = data.get('structures')
    other_structure = data.get('other_structure')
    input_size = other_structure.get('Input_size')
    for structure in structures:
        output_size = structure.get('Neuron_num')
        activation = structure.get('Activ_func')
        py_3 += make_linear(input_size, output_size)
        py_3 += make_activ(activation)
        input_size = output_size
    output_size = other_structure.get('Output_size')
    py_3 += make_linear(input_size, output_size)

    python_code = py_1 + py_2 + py_3 + py_4 + py_5
    file_name = './python/model_config.py'
    with open(file_name, 'w') as file:
        file.write(python_code)
    print('保存完了')