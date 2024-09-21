# 必要なライブラリのインポート
import os
import tempfile
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
import importlib.util
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from flask_socketio import emit
from utils.get_func import get_optimizer, get_loss
from utils.db_manage import download_file, upload_file, initialize_training_results, upload_training_result
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import json
import math
import cv2
import base64

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ImageClassification:{device}")

# GCN関数
class GCN():
    def __init__(self):
        pass

    def __call__(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        return (x - mean)/(std + 10**(-6))  # 0除算を防ぐ

# ZCA白色化の実装
class ZCAWhitening():
    def __init__(self, epsilon=1e-4, device="cuda"):  # 計算が重いのでGPUを用いる
        self.epsilon = epsilon
        self.device = device
        self.mean = None
        self.ZCA_matrix = None

    def fit(self, images):  # 変換行列と平均をデータから計算
        """
        Argument
        --------
        images : torchvision.datasets.cifar.CIFAR10
            入力画像（訓練データ全体）．(N, C, H, W)
        """
        x = images[0][0].reshape(1, -1)  # 画像（1枚）を1次元化
        self.mean = torch.zeros([1, x.size()[1]]).to(self.device)  # 平均値を格納するテンソル．xと同じ形状
        con_matrix = torch.zeros([x.size()[1], x.size()[1]]).to(self.device)
        for i in range(len(images)):  # 各データについての平均を取る
            x = images[i][0].reshape(1, -1).to(self.device)
            self.mean += x / len(images)
            con_matrix += torch.mm(x.t(), x) / len(images)
            if i % 10000 == 0:
                print("{0}/{1}".format(i, len(images)))
        con_matrix -= torch.mm(self.mean.t(), self.mean)
        # E: 固有値 V: 固有ベクトルを並べたもの
        E, V = torch.linalg.eigh(con_matrix)  # 固有値分解
        self.ZCA_matrix = torch.mm(torch.mm(V, torch.diag((E.squeeze()+self.epsilon)**(-0.5))), V.t())  # A(\Lambda + \epsilon I)^{-1/2}A^T
        print("completed!")

    def __call__(self, x):
        size = x.size()
        x = x.reshape(1, -1).to(self.device)
        x -= self.mean  # x - \bar{x}
        x = torch.mm(x, self.ZCA_matrix.t())
        x = x.reshape(tuple(size))
        x = x.to("cpu")
        return x
    
    @classmethod
    def load(cls, filepath):
        state = torch.load(filepath, weights_only=True)
        instance = cls(epsilon=state['epsilon'], device=state['device'])
        instance.mean = state['mean']
        instance.ZCA_matrix = state['ZCA_matrix']
        return instance

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

# カスタムデータセット
class CustomDataset(Dataset):
    def __init__(self, x_train, t_train, transform=None):
        data = x_train.astype('float32')
        # self.x_train = data
        # data = np.transpose(x_train, (0, 2, 3, 1)).astype('float32')
        self.x_train = []
        if x_train.shape[3] == 3:
            for i in range(data.shape[0]):
                self.x_train.append(Image.fromarray(np.uint8(data[i])))
        else:
            for i in range(x_train.shape[0]):
                # グレースケール画像を2D配列として扱う
                img = x_train[i].squeeze()  # (28, 28, 1) -> (28, 28)
                # 0-255の範囲にスケーリング（必要な場合）
                # img = (img * 255).astype(np.uint8)
                self.x_train.append(Image.fromarray(img, mode='L'))
        self.t_train = t_train
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, idx):
        x_train = self.transform(self.x_train[idx])
        t_train = torch.tensor(self.t_train[idx], dtype=torch.long)

        return x_train, t_train

# データセットの読込み＆前処理を行う関数
def load_and_split_data(config):
    image_shape = int(config['input_info']['change_shape'])
    transform_list = [
        transforms.Resize((image_shape, image_shape)),
        transforms.ToTensor()
    ]
    project_name = config["project_name"]
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "./dataset", project_name))
    
    x_train = np.load(os.path.join(dataset_dir, "x_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
    x_test = np.load(os.path.join(dataset_dir, "x_test.npy"))
    y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))

    pretreatment = config['input_info'].get("preprocessing")
    if pretreatment == "GCN":
        print(f'{pretreatment}使用')
        gcn = GCN()
        transform_list.append(gcn)
    elif pretreatment == "ZCA":
        print(f'{pretreatment}使用')
        zca = ZCAWhitening.load(os.path.join(dataset_dir, f"{project_name}_zca.pth"))
        transform_list.append(zca)
    
    test_size = float(config["Train_info"]["test_size"])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size)

    base_transform = transforms.Compose(transform_list)

    # image_shape = config["Train_info"]["image_shape"]
    # if len(x_train.shape) == 2 and x_train.shape[1] == image_shape * image_shape:
    #     channels = 1
    #     height, width = image_shape, image_shape
    #     x_train = x_train.reshape(-1, channels, height, width)
    #     x_val = x_val.reshape(-1, channels, height, width)
    #     x_test = x_test.reshape(-1, channels, height, width)
    # elif len(x_train.shape) == 4:
    #     channels = x_train.shape[-1]
    #     height, width = x_train.shape[1], x_train.shape[2]
    #     x_train = x_train.transpose(0, 3, 1, 2)
    #     x_val = x_val.transpose(0, 3, 1, 2)
    #     x_test = x_test.transpose(0, 3, 1, 2)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), base_transform

# データ拡張のtransformを生成する関数
def create_augmentation_transform(augmentation_params, base_transform):
    # base_transformを先に適用
    augmentation_transforms = base_transform.transforms.copy()
    
    # データ拡張のトランスフォームを追加
    # 回転
    if "rotation_degrees" in augmentation_params and augmentation_params["rotation_degrees"] > 0:
        degrees = augmentation_params["rotation_degrees"]
        augmentation_transforms.append(transforms.RandomRotation(degrees=(-degrees, degrees)))
    
    # 平行移動
    if "vertical_translation_factor" in augmentation_params and "horizontal_translation_factor" in augmentation_params:
        h_trans = augmentation_params["horizontal_translation_factor"]
        v_trans = augmentation_params["vertical_translation_factor"]
        if h_trans > 0 or v_trans > 0:
            augmentation_transforms.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(h_trans, v_trans)
                )
            )
    
    # スケーリング
    if "scaling_factor" in augmentation_params and augmentation_params["scaling_factor"] > 0:
        scale_factor = augmentation_params["scaling_factor"]
        min_scale = max(0.1, 1.0 - scale_factor)
        max_scale = 1.0 + scale_factor
        augmentation_transforms.append(
            transforms.RandomAffine(
                degrees=0,
                scale=(min_scale, max_scale)
            )
        )
    
    # ズーム
    if "zoom_factor" in augmentation_params and augmentation_params["zoom_factor"] > 0:
        zoom_factor = augmentation_params["zoom_factor"]
        min_zoom = max(0.1, 1.0 - zoom_factor)
        max_zoom = 1.0 + zoom_factor
        augmentation_transforms.append(
            transforms.RandomResizedCrop(
                size=base_transform.transforms[0].size,
                scale=(min_zoom, max_zoom)
            )
        )
    
    # 明るさ
    color_jitter_params = {}
    if "brightness_factor" in augmentation_params and augmentation_params["brightness_factor"] > 0:
        brightness = augmentation_params["brightness_factor"]
        color_jitter_params["brightness"] = (max(0, 1.0 - brightness), 1.0 + brightness)
    # コントラスト
    if "contrast_factor" in augmentation_params and augmentation_params["contrast_factor"] > 0:
        contrast = augmentation_params["contrast_factor"]
        color_jitter_params["contrast"] = (max(0, 1.0 - contrast), 1.0 + contrast)
    # 彩度
    if "saturation_factor" in augmentation_params and augmentation_params["saturation_factor"] > 0:
        saturation = augmentation_params["saturation_factor"]
        color_jitter_params["saturation"] = (max(0, 1.0 - saturation), 1.0 + saturation)
    # 色相
    if "hue_factor" in augmentation_params and augmentation_params["hue_factor"] > 0:
        hue = augmentation_params["hue_factor"]
        color_jitter_params["hue"] = (-hue, hue)
    if color_jitter_params:
        augmentation_transforms.append(transforms.ColorJitter(**color_jitter_params))
    
    # シャープネス
    if "sharpness_factor" in augmentation_params and augmentation_params["sharpness_factor"] > 0:
        sharpness = augmentation_params["sharpness_factor"]
        augmentation_transforms.append(transforms.RandomAdjustSharpness(sharpness_factor=sharpness, p=0.5))
        
    # せん断（shear）
    if "shear_angle" in augmentation_params and int(augmentation_params["shear_angle"]) > 0:
        shear_angle = int(augmentation_params["shear_angle"])
        augmentation_transforms.append(
            transforms.RandomAffine(
                degrees=0,
                shear=shear_angle
            )
        )
    
    # ノイズ追加
    if "noise_factor" in augmentation_params and augmentation_params["noise_factor"] > 0:
        noise_factor = augmentation_params["noise_factor"]
        augmentation_transforms.append(transforms.Lambda(lambda img: img + torch.randn_like(img) * noise_factor))
    
    # 水平反転
    if augmentation_params.get("do_flipping", False):
        augmentation_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # 垂直反転
    if augmentation_params.get("do_vertical_flipping", False):
        augmentation_transforms.append(transforms.RandomVerticalFlip(p=0.5))
    
    # グレースケール
    if "grayscale_p" in augmentation_params and augmentation_params["grayscale_p"] > 0:
        grayscale_p = augmentation_params["grayscale_p"]
        augmentation_transforms.append(transforms.RandomGrayscale(p=grayscale_p))
    
    # Augmentationを適用したtransformを作成
    transform = transforms.Compose(augmentation_transforms)
    return transform

# grad-camの実装
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = []
        self.activations = None

        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        # Register hooks only for the specified target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))

    def generate_cam(self, input_image, class_idx):
        self.gradients = []
        self.model.eval()

        # Forward pass
        output = self.model(input_image)
        print(output)
        self.model.zero_grad()

        # Backward pass
        target = torch.zeros_like(output)
        target[0][class_idx] = 1
        output.backward(gradient=target)

        # Compute weights
        gradients = self.gradients[0]
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = cam.squeeze().cpu().detach().numpy()
        return cam

    def __del__(self):
        for hook in self.hooks:
            hook.remove()

def to_heatmap(x, height, width):
    x = (x * 255).astype(np.uint8).reshape(-1)
    cm = plt.get_cmap('jet')
    x = np.array([cm(int(np.round(xi)))[:3] for xi in x])
    print(int(math.sqrt(x.shape[0])))
    return np.resize(x.reshape(int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0])), 3), (height, width, 3))

def visualize_cam(cam, original_image, alpha=0.5, title='Grad-CAM'):
    # Determine the height and width from the original image
    _, height, width = original_image.squeeze(0).shape

    # Convert the CAM to a heatmap
    heatmap = to_heatmap(cam, height, width)

    # Convert original image tensor to numpy for visualization
    original_image_np = original_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Combine original image with heatmap
    grad_cam_image = original_image_np * alpha + heatmap * (1 - alpha)

    # Normalize for display
    grad_cam_image = grad_cam_image / grad_cam_image.max()

    return grad_cam_image

# モデルの訓練を行う関数
def train_model(config):
    model_id = config["model_id"]
    user_id = config["user_id"]
    project_name = config["project_name"]
    augmentation_params = config["augmentation_params"]
    model = import_model(config).to(device)
    with open(f'./dataset/{project_name}/config.json', 'r') as jsonfile:
        json_data = json.load(jsonfile)
        id2label = json_data['id2label']

    # データセットのロードと分割
    (x_train, y_train), (x_val, y_val), (x_test, y_test), base_transform = load_and_split_data(config)

    train_info = config["Train_info"]

    # データ拡張のtransformを生成
    train_transform = create_augmentation_transform(augmentation_params, base_transform)

    # データセットの作成
    train_dataset = CustomDataset(x_train, y_train, train_transform)
    val_dataset = CustomDataset(x_val, y_val, base_transform)       # 検証データにはデータ拡張は適用しない
    test_dataset = CustomDataset(x_test, y_test, base_transform)    # テストデータにもデータ拡張は適用しない

    train_loader = DataLoader(train_dataset, batch_size=int(train_info["batch"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(train_info["batch"]), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    optimizer = get_optimizer(train_info["optimizer"], model.parameters(), float(train_info["learning_rate"]))
    loss_fn = get_loss('cross_entropy')

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model = None
    
    init_result = initialize_training_results(model_id, "ImageClassification")
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

        # 5epochごとの検証
        origin_image_list = []
        label_list = []
        pre_label_list = []
        if epoch % 1 == 0:
            with torch.no_grad():
                c = 0
                for image, label in test_loader:
                    c += 1
                    np_image = image.squeeze(0).permute(1, 2, 0).numpy()
                    _, origin_img_png = cv2.imencode('.png', (np_image * 255).astype(np.uint8))
                    img_base64 = base64.b64encode(origin_img_png).decode()
                    # データを入れる
                    origin_image_list.append(img_base64)
                    label_str = id2label.get(str(label.cpu().item()))
                    label_list.append(label_str)
                    # 予測
                    image, label = image.to(device), label.to(device)
                    outputs = model(image)
                    # 予測結果の確認
                    output_label = torch.max(outputs, 1)[1]
                    pre_label = id2label.get(str(output_label.cpu().item()))
                    pre_label_list.append(pre_label)
                    if c == 10:
                        break
                    
            # emitでデータを渡す
            sent_data = {
                'Epoch': epoch,
                'Images': origin_image_list,
                'Labels': label_list,
                'PreLabels': pre_label_list
            }
            # print(sent_data)
            emit('image_valid'+str(model_id), sent_data)

        # Firestoreに結果をアップロード
        upload_result = upload_training_result(config["user_id"], config["project_name"], model_id, epoch_result)
        # print(upload_result)
        
        print('Epoch:', epoch, 'TrainAcc:', round(epoch_acc, 5), 'ValAcc:', round(val_acc, 5), 'TrainLoss:', round(epoch_loss, 5), 'ValLoss:', round(val_loss, 5))
        emit('train_image_results'+model_id, epoch_result)
        
        if val_loss < best_val_loss:
            best_val_loss = round(val_loss, 5)
            best_val_acc = round(val_acc, 5)
            best_model = model.state_dict()
    
    running_test_loss = 0
    running_test_corrects = 0
    # テスト
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_test_loss += loss.item()
            running_test_corrects += (torch.max(outputs, 1)[1] == targets).sum().item()
    test_loss = round(running_test_loss / len(test_loader), 5)
    test_acc = round(running_test_corrects / len(test_loader.dataset), 5)
    # 一時ファイルにモデルを保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        best_model_path = tmp_file.name
        torch.save(model.state_dict(), best_model_path)
    
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
    
    return test_acc, test_loss
