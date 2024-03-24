# pytorch
import torch
import torch.nn as nn

# ライブラリ
import numpy as np
from collections import deque
import random

# 自分が用意したライブラリ
from utils.select import get_activation, get_optimizer, get_loss


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        # buffer_sizeのキューを作成
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data])
        return state, action, reward, next_state, done


class Simple_Network(nn.Module):
    def __init__(self, structures, other_structure):
        super().__init__()
        input_size = other_structure.get('Input_size')
        output_size = other_structure.get('Output_size')
        layer_list = []
        for structure in structures:
            neuron_num = structure.get('Neuron_num')
            activ_func = structure.get('Activ_func')
            layer_n = nn.Linear(int(input_size), int(neuron_num))
            layer_a = get_activation(activ_func)
            layer_list.append(layer_n)
            layer_list.append(layer_a)
            input_size = neuron_num
        layer_n = nn.Linear(int(input_size), int(output_size))
        layer_list.append(layer_n)
        self.model = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.model(x)


class CNN_Network(nn.Module):
    def __init__(self, structures, other_structure):
        super().__init__()
        self.input_size_list = other_structure.get('Input_size')
        output_size = int(other_structure.get('Output_size'))
        cnn_layer_list = []
        Conv_Layer = structures.get('Conv_Layer')
        for CL_structure in Conv_Layer:
            layer_name = CL_structure.get('Layer_name')
            if layer_name == 'Conv2d':
                in_channel = int(CL_structure.get('In_channel'))
                out_channel = int(CL_structure.get('Out_channel'))
                kernel_size = int(CL_structure.get('Kernel_size'))
                stride = int(CL_structure.get('Stride'))
                padding = int(CL_structure.get('Padding'))
                active_func = CL_structure.get('Active_func')
                conv_layer = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
                activ_layer = get_activation(active_func)
                cnn_layer_list.append(conv_layer)
                cnn_layer_list.append(activ_layer)
            elif layer_name == 'MaxPool2d':
                kernel_size = int(CL_structure.get('Kernel_size'))
                stride = int(CL_structure.get('Stride'))
                padding = int(CL_structure.get('Padding'))
                pool_layer = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
                cnn_layer_list.append(pool_layer)
        self.features = nn.Sequential(*cnn_layer_list)

        conv_output_size = self._calculate_conv_output_size()

        Connected_Layer = structures.get('Fully_Connected_Layer')
        connected_layer_list = []
        if len(Connected_Layer) > 0:
            input_size = conv_output_size
            for C_layer in Connected_Layer:
                neuron_num = int(C_layer.get('Neuron_num'))
                activ_func = C_layer.get('Activ_func')
                l_layer = nn.Linear(input_size, neuron_num)
                a_layer = get_activation(activation_name=activ_func)
                connected_layer_list.append(l_layer)
                connected_layer_list.append(a_layer)
                input_size = neuron_num
            l_layer = nn.Linear(input_size, output_size)
            connected_layer_list.append(l_layer)
        else:
            input_size = conv_output_size
            l_layer = nn.Linear(input_size, output_size)
            connected_layer_list.append(l_layer)
        self.fc = nn.Sequential(*connected_layer_list)
    
    def _calculate_conv_output_size(self):
        H = int(self.input_size_list[0])
        W = int(self.input_size_list[1])
        C = int(self.input_size_list[2])
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)  # Batch size of 1, 4 channels, 80x80 image
            conv_output = self.features(dummy_input)
            flattened_size = conv_output.view(1, -1).size(1)
        return flattened_size
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DQNAgent:
    def __init__(self, train_info, structures, other_structure, QNet_model, device='cpu'):
        # デバイスの設定(引数が無ければ自動でCPUになる)
        self.device = device

        # Q値計算用
        self.gamma = 0.9
        self.lr = float(train_info.get('Learning_rate'))
        self.epsilon = float(train_info.get('Epsilon'))
        self.buffer_size = int(train_info.get('Buffer_size'))
        self.batch_size = int(train_info.get('Batch_num'))
        self.action_size = int(train_info.get('Action_size'))

        self.loss_func = get_loss(train_info.get('Loss'))

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet_model().to(device=self.device)
        self.qnet_target = QNet_model().to(device=self.device)
        self.optimizer = get_optimizer(train_info.get('Optimizer'), self.qnet.parameters(), self.lr)
        # 勾配を0で初期化する(学習時は毎回行う)
        self.optimizer.zero_grad()
    
    # ネットワークの同期
    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())
    
    # 行動の決め方
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            state = torch.tensor(state, dtype=torch.float32).to(device=self.device)
            # qs = self.qnet_target.forward(state)
            qs = self.qnet.forward(state)
            # print(f'qs:{qs}, action:{int(torch.argmax(qs))}')
            return int(torch.argmax(qs))
    
    # Q関数の更新
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        state = torch.tensor(state, dtype=torch.float32).to(device=self.device)
        qs = self.qnet.forward(state)
        q = qs[torch.arange(self.batch_size), action]

        next_state = torch.tensor(next_state, dtype=torch.float32).to(device=self.device)
        next_qs = self.qnet_target.forward(next_state)
        next_q, index = next_qs.max(dim=1)
        # next_q.detach()
        reward = torch.tensor(reward, dtype=torch.float32).to(device=self.device)
        done = torch.tensor(done, dtype=torch.float32).to(device=self.device)
        target = reward + (1 - done) * self.gamma * next_q
        targets = self.qnet.forward(state)
        new_targets = []  # 新しいターゲットを格納するリストを作成

        for ts, t, i in zip(targets, target, index):
            new_ts = ts.clone()  # ビューではない新しいテンソルを作成
            new_ts[i] = t  # 新しいテンソルに値を代入
            new_targets.append(new_ts)  # 新しいテンソルをリストに追加

        # リストをPyTorchテンソルに変換し、元のtargetsを置き換える
        targets = torch.stack(new_targets).to(device=self.device)
        # reward = reward.reshape(32, 1)
        # done = done.reshape(32, 1)
        # targets = reward + (1 - done) * self.gamma * next_qs

        loss = self.loss_func(q, target)
        # loss = F.huber_loss(qs, targets)
        

        # 勾配を0で初期化
        self.optimizer.zero_grad()
        # 逆伝播
        loss.backward()
        # 重みの更新
        self.optimizer.step()

        # if int(loss) == 0:
        #     print(int(loss))
        #     print(f'qs{qs}')
        #     print(f'targets{targets}')

        return loss


class TryAgent:
    def __init__(self, QNet_model):
        self.qnet = QNet_model().to(device='cpu')
        self.qnet.load_state_dict(torch.load('./save/best_model.pth'))
    
    def get_action(self, state):
        state = state[np.newaxis, :]
        state = torch.tensor(state, dtype=torch.float32).to(device='cpu')
        # qs = self.qnet_target.forward(state)
        qs = self.qnet.forward(state)
        # print(f'qs:{qs}, action:{int(torch.argmax(qs))}')
        return int(torch.argmax(qs))


if __name__ == '__main__':
    li = []
    print(len(li))