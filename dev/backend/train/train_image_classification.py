import os
import tempfile
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import importlib.util
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from flask_socketio import emit
from utils.get_func import get_optimizer, get_loss
from utils.db_manage import download_file, upload_file, initialize_training_results, upload_training_result

import matplotlib
matplotlib.use('Agg')

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ImageClassification:{device}")

# GCN関数
def gcn(x):
    mean = np.mean(x, axis=(1,2,3), keepdims=True)
    std = np.std(x, axis=(1,2,3), keepdims=True)
    return (x - mean) / (std + 1.E-6)

# 新しいZCA Whiteningクラス
class ZCA_Whitening:
    def __init__(self, epsilon=1E-6):
        self.epsilon = epsilon
        self.mean = None
        self.PCA_mat = None
        
    def fit(self, x):
        x = x.astype(np.float64)
        x = x.reshape(x.shape[0],-1)
        self.mean = np.mean(x, axis=0)
        x -= self.mean
        cov_mat = np.dot(x.T, x) / x.shape[0]
        A, L, _ = np.linalg.svd(cov_mat)
        self.ZCA_mat = np.dot(A, np.dot(np.diag(1. / (np.sqrt(L) + self.epsilon)), A.T))
            
    def transform(self, x):
        shape = x.shape
        x = x.astype(np.float64)
        x = x.reshape(x.shape[0],-1)
        x -= self.mean
        x = np.dot(x, self.ZCA_mat)
        return x.reshape(shape)

# モデルをインポートする関数
def import_model(config):
    user_id = config["user_id"]
    project_name = config["project_name"]
    model_id = config["model_id"]
    
    model_blob_path = f"user/{user_id}/{project_name}/{model_id}/model_config.py"
    
    # 一時ファイルの作成とダウンロード
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp_file:
        model_local_path = download_file(model_blob_path, tmp_file.name)
        tmp_file.close()
    
    if model_local_path is None:
        raise FileNotFoundError(f"Model file not found in Firebase Storage: {model_blob_path}")

    # ダウンロードされたファイルの内容を確認
    with open(model_local_path, 'r') as file:
        content = file.read()
        print(f"Downloaded file content:\n{content}")

    # モジュールのインポートとエラーハンドリング
    try:
        spec = importlib.util.spec_from_file_location("Simple_NN", model_local_path)
        if spec is None:
            raise ImportError(f"Could not load spec from file: {model_local_path}")
        model_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError(f"Could not load loader from spec: {spec}")
        spec.loader.exec_module(model_module)
        return model_module.Simple_NN()
    except Exception as e:
        raise ImportError(f"Failed to import model: {e}")

# データセットの読込み＆前処理を行う関数
def load_and_split_data(config):
    project_name = config["project_name"]
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "./dataset", project_name))
    
    x_train = np.load(os.path.join(dataset_dir, "x_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
    x_test = np.load(os.path.join(dataset_dir, "x_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))

    pretreatment = config["Train_info"].get("Pretreatment", "none")
    if pretreatment == "GCN":
        x_train = gcn(x_train)
        x_test = gcn(x_test)
    elif pretreatment == "ZCA":
        zca = ZCA_Whitening()
        zca.fit(x_train)
        x_train = zca.transform(x_train)
        x_test = zca.transform(x_test)
    
    test_size = float(config["Train_info"]["test_size"])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size)

    image_shape = config["Train_info"]["image_shape"]
    if len(x_train.shape) == 2 and x_train.shape[1] == image_shape * image_shape:
        channels = 1
        height, width = image_shape, image_shape
        x_train = x_train.reshape(-1, channels, height, width)
        x_val = x_val.reshape(-1, channels, height, width)
        x_test = x_test.reshape(-1, channels, height, width)
    elif len(x_train.shape) == 4:
        channels = x_train.shape[-1]
        height, width = x_train.shape[1], x_train.shape[2]
        x_train = x_train.transpose(0, 3, 1, 2)
        x_val = x_val.transpose(0, 3, 1, 2)
        x_test = x_test.transpose(0, 3, 1, 2)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# モデルの訓練を行う関数
def train_model(config):
    model_id = config["model_id"]
    user_id = config["user_id"]
    project_name = config["project_name"]
    model = import_model(config).to(device)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_data(config)

    train_info = config["Train_info"]

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
    
    train_loader = DataLoader(train_dataset, batch_size=int(train_info["batch"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(train_info["batch"]), shuffle=False)

    optimizer = get_optimizer(train_info["optimizer"], model.parameters(), float(train_info["learning_rate"]))
    loss_fn = get_loss('cross_entropy')

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model = None
    
    init_result = initialize_training_results(user_id, project_name, model_id)
    print(f"DB初期化:{init_result}")

    print('学習スタート')
    for epoch in range(1, int(train_info["epoch"])+1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
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
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_val_loss += loss.item()
                running_val_corrects += (torch.max(outputs, 1)[1] == targets).sum().item()
        
        val_loss = running_val_loss / len(val_loader)
        val_acc = running_val_corrects / len(val_loader.dataset)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        # エポックごとの結果を辞書に格納
        epoch_result = {
            'Epoch': epoch,
            'TrainAcc': round(epoch_acc, 5),
            'ValAcc': round(val_acc, 5),
            'TrainLoss': round(epoch_loss, 5),
            'ValLoss': round(val_loss, 5)
        }
        
        # Firestoreに結果をアップロード
        upload_result = upload_training_result(config["user_id"], config["project_name"], model_id, epoch_result)
        print(upload_result)
        
        print('Epoch:', epoch, 'TrainAcc:', round(epoch_acc, 5), 'ValAcc:', round(val_acc, 5), 'TrainLoss:', round(epoch_loss, 5), 'ValLoss:', round(val_loss, 5))
        emit('train_image_results'+model_id, epoch_result)
        
        if val_loss < best_val_loss:
            best_val_loss = round(val_loss, 5)
            best_val_acc = round(val_acc, 5)
            best_model = model.state_dict()

    # 一時ファイルにモデルを保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        best_model_path = tmp_file.name
        torch.save(best_model, best_model_path)
    
    # Firebase Storageにモデルをアップロード
    model_storage_path = f"user/{user_id}/{project_name}/{model_id}/best_model.pth"
    upload_result = upload_file(best_model_path, model_storage_path)
    print(upload_result)
    
    # 画像を保存してFirebase Storageにアップロード
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        accuracy_curve_path = tmp_file.name
        plt.figure()
        plt.title("Training Accuracy")
        plt.plot(range(1, int(train_info["epoch"])+1), train_acc_history, label="Train Accuracy")
        plt.plot(range(1, int(train_info["epoch"])+1), val_acc_history, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(accuracy_curve_path)
        plt.close()
        accuracy_curve_storage_path = f"user/{user_id}/{project_name}/{model_id}/accuracy_curve.png"
        upload_file(accuracy_curve_path, accuracy_curve_storage_path)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        loss_curve_path = tmp_file.name
        plt.figure()
        plt.title('Training Loss')
        plt.plot(range(1, int(train_info["epoch"])+1), train_loss_history, label="Train Loss")
        plt.plot(range(1, int(train_info["epoch"])+1), val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(loss_curve_path)
        plt.close()
        loss_curve_storage_path = f"user/{user_id}/{project_name}/{model_id}/loss_curve.png"
        upload_file(loss_curve_path, loss_curve_storage_path)
    
    return best_val_acc, best_val_loss
