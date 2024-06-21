import gyms.FlappyBird.game.wrapped_flappy_bird as game

import torch

import numpy as np
import cv2
from flask_socketio import emit
import base64

from utils.gym_utils import DQNAgent, CNN_Network
from utils.resize_image import get_make_image


def make_show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (1, 0, 2))
    return image


def flappybird(structures, other_structure, train_info, id):
    agent = DQNAgent(train_info, structures, other_structure, CNN_Network, device='cuda:0')
    episodes = int(train_info.get('Epoch'))
    sync_interval = 20
    max_reward = 0
    env = game.GameState()

    input_size_list = other_structure.get('Input_size')
    H = int(input_size_list[0])
    W = int(input_size_list[1])
    C = int(input_size_list[2])
    image_make_func = get_make_image(C)

    max_reward = 0

    for episode in range(episodes):
        env.__init__()
        action_array = np.zeros(2)
        action_array[0] = 1
        image, reward, terminal = env.frame_step(action_array)
        train_image = image_make_func(image, H, W)
        show_image = make_show_image(image)

        total_reward = 0
        total_loss = 0
        cut = 0

        while not terminal:
            cut += 1
            action_array = np.zeros(2)
            action_index = agent.get_action(train_image)
            action_array[action_index] = 1
            image, reward, terminal = env.frame_step(action_array)
            next_train_image = image_make_func(image, H, W)
            show_image = make_show_image(image)
            _, buffer = cv2.imencode('.png', show_image)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            if terminal:
                next_train_image = torch.zeros_like(next_train_image)
            
            # ソケット通信
            emit('FlappyBird_data'+id, {'image_data': encoded_image})

            total_loss += agent.update(train_image, action_index, reward, next_train_image, terminal)
            train_image = next_train_image
            total_reward += reward
            
        if episode % sync_interval == 0:
            agent.sync_qnet()
        
        if max_reward < total_reward:
            max_reward = total_reward

        print(f'episode:{episode} loss_average:{total_loss/cut} reward_average:{total_reward/cut} total_reward:{total_reward}')

    env.close()
    print('終わり')
