import torch
import torch.nn as nn
from utils.select_activate import get_activation

class QNet_model(nn.Module):
    def __init__(self, structures, other_structure):
        super().__init__()
        input_size = other_structure.get('input_size')
        output_size = other_structure.get('output_size')
        self.layer_list = []
        for structure in structures:
            neuron_num = structure.get('Neuron_num')
            activ_func = structure.get('Activ_func')
            layer_n = nn.Linear(input_size, neuron_num)
            layer_a = get_activation(activ_func)
            self.layer_list.append(layer_n)
            self.layer_list.append(layer_a)
            input_size = neuron_num
        layer_n = nn.Linear(input_size, output_size)
        self.layer_list.append(layer_n)
    
    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

def get_model(structures, other_structure):
    # structure: list
    #   - Neuron_num: int
    #   - Activ_func: str
    model = QNet_model(structures, other_structure)
    return model