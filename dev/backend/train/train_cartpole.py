import os
import importlib
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from collections import deque
import random
import cv2
from flask_socketio import emit
from utils.get_func import get_optimizer, get_loss
from train.train_image_classification import import_model
from utils.db_manage import upload_training_result, upload_file, initialize_training_results
import matplotlib
matplotlib.use('Agg')

# デバイスを指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CartPole:{device}")

# リプレイバッファクラスの定義
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
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

# DQNエージェントクラスの定義
class DQNAgent:
    def __init__(self, train_info, config, device=device):
        self.device = device
        self.gamma = 0.9  # 割引率
        self.lr = float(train_info['learning_rate'])
        self.epsilon = float(train_info['episilon'])
        self.buffer_size = int(train_info['buffer'])
        self.batch_size = int(train_info['batch'])
        self.action_size = 2
        self.loss_func = get_loss('mse_loss')
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = import_model(config=config).to(device=self.device)
        self.qnet_target = import_model(config=config).to(device=self.device)
        self.optimizer = get_optimizer(train_info['optimizer'], self.qnet.parameters(), self.lr)
        self.optimizer.zero_grad()
    
    # ターゲットネットワークを同期する関数
    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())
    
    # 行動を選択する関数
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(device=self.device)
            qs = self.qnet.forward(state)
            return int(torch.argmax(qs))
    
    # Qネットワークを更新する関数
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return 0
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        qs = self.qnet.forward(state)
        q = qs[torch.arange(self.batch_size), action]
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        next_qs = self.qnet_target.forward(next_state)
        next_q, index = next_qs.max(dim=1)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)
        target = reward + (1 - done) * self.gamma * next_q
        targets = self.qnet.forward(state)
        # new_targets = [ts.clone().index_fill_(0, torch.tensor([i]), t) for ts, t, i in zip(targets, target, index)]
        new_targets = [ts.clone().index_fill_(0, torch.tensor([i], device=device), t.to(device)) for ts, t, i in zip(targets, target, index)]
        targets = torch.stack(new_targets).to(self.device)
        loss = self.loss_func(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# モデルを訓練する関数
def train_cartpole(config, socketio):
    user_id = config["user_id"]
    project_name = config["project_name"]
    model_id = config["model_id"]
    train_info = config["Train_info"]
    epoch = train_info["epoch"]
    sync_interval = int(train_info["syns"])
    input_info = config['input_info']
    preprocessing = input_info['preprocessing']
    shape = input_info['change_shape']

    # モデルの取得
    env = gym.make('CartPole-v1')
    agent = DQNAgent(train_info=train_info, config=config, device=device)

    max_reward = 0
    rewards = []
    losses = []

    init_result = initialize_training_results(model_id, "ReinforcementLearning")
    print(init_result)

    for episode in range(1, int(epoch)+1):
        state, _ = env.reset()
        state = cv2.resize(state, (shape, shape))
        if preprocessing == 'GRAY':
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            ret, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
            state = np.reshape(state, (shape, shape, 1))
        done = False
        total_reward = 0
        total_loss = 0
        step = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = cv2.resize(next_state, (shape, shape))
            if preprocessing == 'GRAY':
                next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
                ret, next_state = cv2.threshold(next_state, 1, 255, cv2.THRESH_BINARY)
                next_state = np.reshape(next_state, (shape, shape, 1))
            done = done or truncated
            loss = agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            total_loss += loss
            step += 1
        ave_loss = total_loss / step
        rewards.append(total_reward)
        losses.append(ave_loss)

        if episode % sync_interval == 0:
            agent.sync_qnet()

        # エポックごとの結果を辞書に格納
        epoch_result = {
            "Epoch": episode,
            "TotalReward": total_reward,
            "AverageLoss": round(ave_loss, 5)
        }

        # firebaseに結果を格納
        upload_result = upload_training_result(user_id=user_id, project_name=project_name, model_id=model_id, epoch_result=epoch_result)
        print(upload_result)

        # socketio.sleep(0.15)
        print(f'Epoch: {episode} TotalReward: {round(total_reward, 5)} AverageLoss: {round(ave_loss, 5)}')
        emit('train_cortpole_results'+model_id, epoch_result)

    # 一時ファイルにモデルを保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        best_model_path = tmp_file.name
        torch.save(agent.qnet.state_dict(), best_model_path)
    
    # Firebase Storageにモデルをアップロード
    model_storage_path = f"user/{user_id}/{project_name}/{model_id}/best_model.pth"
    upload_result = upload_file(best_model_path, model_storage_path)
    print(upload_result)

    # 画像を保存してFirebase Storageにアップロード
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        reward_curve_path = tmp_file.name
        plt.figure()
        plt.title("Training Reward")
        plt.plot(range(1, int(train_info["epoch"])+1), rewards, label="Train Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(reward_curve_path)
        plt.close()
        reward_curve_storage_path = f"user/{user_id}/{project_name}/{model_id}/reward_curve.png"
        upload_file(reward_curve_path, reward_curve_storage_path)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        loss_curve_path = tmp_file.name
        plt.figure()
        plt.title('Training Loss')
        plt.plot(range(1, int(train_info["epoch"])+1), losses, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(loss_curve_path)
        plt.close()
        loss_curve_storage_path = f"user/{user_id}/{project_name}/{model_id}/loss_curve.png"
        upload_file(loss_curve_path, loss_curve_storage_path)
    
    return round(total_reward, 5), round(ave_loss, 5)
