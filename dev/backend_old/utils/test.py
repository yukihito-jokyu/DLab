import gym
import numpy as np
from flask_socketio import emit

def test_env():
    env = gym.make('CartPole-v1', render_mode='human')

    state = env.reset()
    print(state)
    done = False
    s = 0
    while True:
        env.render()
        action = np.random.choice([0, 1])
        next_state, reward, done, info, i = env.step(action)
        print(next_state)
        location = float(next_state[0])
        radian = float(next_state[2])
        s += 1
        emit('CartPole_data', {'location': location, 'radian': radian})
        if s > 100:
            break

    env.close()