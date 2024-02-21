import gym
import torch
from utils.gym_utils import get_model
from flask_socketio import emit


def cartpole(structures, other_structure):
    # モデルの作成
    # 学習の詳細情報
    model = get_model(structures, other_structure)