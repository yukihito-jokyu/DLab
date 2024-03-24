from utils.gym_utils import TryAgent
from save.model_config import Simple_NN

import gym
from flask_socketio import emit

import numpy as np

def trycartpole():
    env = gym.make('CartPole-v1', render_mode='human')
    agent = TryAgent(Simple_NN)
    episodes = 300

    for episode in range(episodes):
        state = env.reset()[0]
        location = state[0]
        radian = state[2]
        emit('episode_start', {'episode': episode+1, 'location': float(location), 'radian': float(radian)})
        done = False

        while not done:
            env.render()
            action = agent.get_action(state)
            next_state = env.step(int(action))[0]
            # ソケット通信
            location = next_state[0]
            radian = next_state[2]
            emit('CartPole_data', {'location': float(location), 'radian': float(radian)})

            reward = env.step(int(action))[1]
            done = env.step(int(action))[2]

            if done:
                next_state = np.zeros(state.shape)
            state = next_state
    env.close()
    emit('end', {'message': 'Processing complete!'})