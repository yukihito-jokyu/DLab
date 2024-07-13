import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import importlib.util
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# GCN関数
def global_contrast_normalization(X, scale=1.0, min_divisor=1e-8):
    mean = X.mean(axis=1, keepdims=True)
    X = X - mean
    contrast = np.sqrt((X**2).sum(axis=1, keepdims=True))
    contrast[contrast < min_divisor] = min_divisor
    X = scale * X / contrast
    return X

# ZCA Whitening関数
def zca_whitening(X):
    X = X.reshape(X.shape[0], -1)
    sigma = np.dot(X.T, X) / X.shape[0]
    U, S, V = np.linalg.svd(sigma)
    epsilon = 1e-5
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    X = np.dot(X, ZCAMatrix)
    return X.reshape(-1, 28, 28)

# データセットを読み込む関数
def import_model(config):
    user_id = config["user_id"]
    project_name = config["Project_name"]
    model_id = config["model_id"]
    
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "./user"))
    model_path = os.path.join(base_dir, user_id, project_name, model_id, "model_config.py")
    spec = importlib.util.spec_from_file_location("Simple_NN", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module.Simple_NN()

# データセットの読込み＆前処理を行う関数
def load_and_split_data(config):
    project_name = config["Project_name"]
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "./dataset", project_name))
    
    x_train = np.load(os.path.join(dataset_dir, "x_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
    x_test = np.load(os.path.join(dataset_dir, "x_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))

    pretreatment = config["Train_info"].get("Pretreatment", "none")
    if pretreatment == "GCN":
        x_train = global_contrast_normalization(x_train)
        x_test = global_contrast_normalization(x_test)
    elif pretreatment == "ZCA":
        x_train = zca_whitening(x_train)
        x_test = zca_whitening(x_test)
    
    test_size = config["Train_info"]["test_size"]
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size)
    
    # 形状を (batch_size, 1, 28, 28) に変換
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_val = x_val.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def get_optimizer(optimizer_name, params, lr):
    if optimizer_name == 'SGD':
        return optim.SGD(params, lr)
    if optimizer_name == 'momentum':
        return optim.SGD(params, lr, momentum=0.8)
    if optimizer_name == 'Adam':
        return optim.Adam(params, lr)
    if optimizer_name == 'Adagrad':
        return optim.Adagrad(params, lr)
    if optimizer_name == 'RMSProp':
        return optim.RMSprop(params, lr)
    if optimizer_name == 'Adadelta':
        return optim.Adadelta(params, lr)

def get_loss(loss_name):
    if loss_name == 'mse_loss':
        return nn.MSELoss()
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    if loss_name == 'binary_cross_entropy':
        return nn.BCELoss()
    if loss_name == 'nll_loss':
        return nn.NLLLoss()
    if loss_name == 'hinge_embedding_loss':
        return nn.HingeEmbeddingLoss()

def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    corrects = (preds == targets).sum().item()
    accuracy = corrects / targets.size(0)
    return accuracy

def train_model(config):
    # モデルのインポート
    model = import_model(config)
    
    # データセットのロードと分割
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_data(config)
    
    # 訓練情報の取得
    train_info = config["Train_info"]
    
    # データローダーの作成
    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
    
    train_loader = DataLoader(train_dataset, batch_size=train_info["batch"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_info["batch"], shuffle=False)
    
    # オプティマイザと損失関数の取得
    optimizer = get_optimizer(train_info["optimizer"], model.parameters(), train_info["learning_rate"])
    loss_fn = get_loss(train_info["loss"])
    
    # トレーニングループ
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(1, train_info["epoch"]+1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_corrects += (torch.max(outputs, 1)[1] == targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_val_loss += loss.item()
                running_val_corrects += (torch.max(outputs, 1)[1] == targets).sum().item()
        
        val_loss = running_val_loss / len(val_loader)
        val_acc = running_val_corrects / len(val_loader.dataset)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        print(f'Epoch [{epoch}/{train_info["epoch"]}], Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    # 最良モデルの保存
    user_id = config["user_id"]
    project_name = config["Project_name"]
    model_id = config["model_id"]
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "./user", user_id, project_name, model_id))
    os.makedirs(base_dir, exist_ok=True)
    best_model_path = os.path.join(base_dir, "best_model.pth")
    torch.save(best_model, best_model_path)
    
    # 学習曲線の保存
    photo_dir = os.path.join(base_dir, "photo")
    os.makedirs(photo_dir, exist_ok=True)
    
    plt.figure()
    plt.title("Training Accuracy")
    plt.plot(range(1, train_info["epoch"]+1), train_acc_history, label="Train Accuracy")
    plt.plot(range(1, train_info["epoch"]+1), val_acc_history, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(photo_dir, "accuracy_curve.png"))
    plt.close()
    
    plt.figure()
    plt.title('Training Loss')
    plt.plot(range(1, train_info["epoch"]+1), train_loss_history, label="Train Loss")
    plt.plot(range(1, train_info["epoch"]+1), val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(photo_dir, "loss_curve.png"))
    plt.close()


if __name__ == '__main__':
    # 実行
    config = {
        "user_id": "example_user_id",
        "Project_name": "MNIST",
        "model_id": "example_model",
        "Train_info": {
            "Pretreatment": "none",
            "loss": "cross_entropy",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch": 16,
            "epoch": 2,
            "test_size": 0.2
        }
    }

    train_model(config)