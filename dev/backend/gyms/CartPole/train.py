import gym
import torch
from utils.gym_utils import DQNAgent
from python.model_config import Simple_NN
from flask_socketio import emit

import numpy as np


def cartpole(structures, other_structure, train_info, id):
    # モデルの作成
    # 学習の詳細情報
    env = gym.make('CartPole-v1', render_mode='human')
    agent = DQNAgent(train_info, structures, other_structure, Simple_NN)
    epoch = int(train_info.get('Epoch'))
    sync_interval = 20
    max_reward = 0

    for episode in range(epoch):
        state = env.reset()[0]
        location = state[0]
        radian = state[2]
        emit('episode_start'+id, {'episode': episode+1, 'location': float(location), 'radian': float(radian)})
        done = False
        total_reward = 0
        total_loss = 0
        cnt = 0

        while not done:
            env.render()
            action = agent.get_action(state)
            next_state = env.step(int(action))[0]
            # ソケット通信
            location = next_state[0]
            radian = next_state[2] 
            emit('CartPole_data'+id, {'location': float(location), 'radian': float(radian)})

            reward = env.step(int(action))[1]
            done = env.step(int(action))[2]

            if done:
                next_state = np.zeros(state.shape)
            
            total_loss += agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            cnt += 1
            if cnt > 200:
                print('200達成')
                break
        
        if episode % sync_interval == 0:
            agent.sync_qnet()
        
        if max_reward < total_reward:
            max_reward = total_reward
            torch.save(agent.qnet.state_dict(), "./weights/best_CartPole.pth")
        
        print(f'loss_average:{total_loss/cnt} reward_average:{total_reward/cnt} total_reward:{total_reward}')
    
    env.close()
    emit('end', {'message': 'Processing complete!'})