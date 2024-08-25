import numpy as np
import json

import cv2
import base64

import os
from torchvision import transforms
from train.train_image_classification import CustomDataset, ZCAWhitening, GCN
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image

class dataset_imager:
    def __init__(self, dataset_name='MNIST', n=100, label='all'):
        self.dataset_name = dataset_name

        # dataset
        self.dataset_path = f'./dataset/{self.dataset_name}/' # データセットのパス
        self.x_filename = ['x_train.npy', 'x_test.npy'] # 画像が記録されているファイル
        self.y_filename = ['y_train.npy', 'y_test.npy'] # ラベルidが記録されているファイル
        with open(self.dataset_path + 'config.json' , 'r') as f:
            self.config = json.load(f) # color : カラータイプ , label2id : ラベル名とラベルidの対応関係 を記録しているファイル
        # query（条件用変数）
        self.n = n
        self.label = label
        # load
        self.x, self.y = self.load_dataset()
    
    def load_dataset(self):
        #
        # データセットを読み込み
        # ------------------------------
        # description :
        #   __init__で指定した条件に従いデータセットを読み込む
        # argmetns :
        #   ※ __init__にて定義
        # return :
        #   x : 画像データ(np.ndarray)
        #   y : ラベルデータ(np.ndarray)
        #

        # label2id（ラベル名とラベルidの対応関係を保存している）
        label2id = self.config['label2id']

        # データの読み込み・結合
        x_shape = list(np.load(self.dataset_path + self.x_filename[0]).shape[1:])
        x_shape.insert(0, 1)
        x = np.array(np.zeros(x_shape)) # 1件のdummyデータを作成
        # flatten_x_shape = np.load(self.dataset_path + self.x_filename[0]).shape[1] # 1次元に保存されている1件の配列数を取得
        # x = np.array(np.zeros((1,flatten_x_shape)))
        for f in self.x_filename:
            add_npy = np.load(self.dataset_path + f)
            x = np.concatenate((x, add_npy), axis=0)
        x = x[1:] # dummyの削除
        x = x.astype(np.uint8) # 画像データはuint8型に変換

        y = np.array(np.zeros((1))) # ラベルはは1次元の配列に保存されている為 [0] を定義
        for f in self.y_filename:
            add_npy = np.load(self.dataset_path + f)
            y = np.concatenate((y, add_npy), axis=0)
        y = y[1:] # dummyの削除

        # シャッフル
        indices = np.arange(len(x)) # x のindexを取得
        np.random.shuffle(indices) # indicesをシャッフル
        # 適応
        x = x[indices]
        y = y[indices]

        # // query //
        # 特定のラベルをチョイス
        if self.label != 'all':
            if self.label in label2id.keys():
                idx = np.where(y == label2id[self.label])[0]
                x = x[idx]
                y = y[idx]
            else:
                x, y = np.array([None]), np.array([None]) # 例外処理
        # データ件数に絞る
        if self.n < len(x):
            x = x[:self.n]
        return x, y

    def dispend_image(self):
        #
        # バイナリコードにする
        # ------------------------------
        # description :
        #   画像データをpng -> base64に変換して返す
        # argmetns :
        #   * __init__にて定義
        # return :
        #   画像データ（base64）(list)
        images = []
        for img in self.x:
            print(img)
            print(img.shape)
            print(type(img))
            if self.config['color'] == 'gray': # グレースケール(チャンネル１)
                img = img.reshape(self.config['shape'])
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # グレースケールを3chに変換
                _, img_png = cv2.imencode('.png', img)
                img_base64 = base64.b64encode(img_png).decode()
                images.append(img_base64)
            if self.config['color'] == 'rgb': # カラー(チャンネル３) * 検証まだ
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _, img_png = cv2.imencode('.png', img)
                img_base64 = base64.b64encode(img_png).decode()
                images.append(img_base64)
        return images


# カスタムデータセット
class PreDataset(Dataset):
    def __init__(self, x_train, y_train, dataset_name, transform=None):
        data = x_train.astype('float32')
        self.label = y_train
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
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.normal_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.dataset_path = f'./dataset/{dataset_name}/'
        with open(self.dataset_path + 'config.json' , 'r') as f:
            self.config = json.load(f)

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, idx):
        normal_x = self.normal_transform(self.x_train[idx])
        pre_x = self.transform(self.x_train[idx])
        label_id = self.label[idx]
        label = self.config['id2label'][str(label_id)]

        return normal_x, pre_x, label

# 標準化後の画像を[0, 1]に正規化する
def deprocess(x):
    """
    Argument
    --------
    x : np.ndarray
        入力画像．(H, W, C)

    Return
    ------
    _x : np.ndarray
        [0, 1]で正規化した画像．(H, W, C)
    """
    _min = np.min(x)
    _max = np.max(x)
    _x = (x - _min)/(_max - _min)
    return _x

# 前処理後の画像の取得
def get_images(config):
    project_name = config['project_name']
    if project_name == 'FlappyBird' or project_name == 'CartPole':
        preprocessing = config['input_info']['preprocessing']
        image_shape = int(config['input_info']['change_shape'])
        dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "./dataset", project_name))
        origin_image = cv2.imread(os.path.join(dataset_dir, "image.png"))
        origin_image = cv2.resize(origin_image, (image_shape, image_shape))
        images = []
        pre_images = []
        _, origin_img_png = cv2.imencode('.png', origin_image)
        img_base64 = base64.b64encode(origin_img_png).decode()
        images.append(img_base64)
        if preprocessing == 'GRAY':
            pre_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
            ret, pre_image = cv2.threshold(pre_image, 1, 255, cv2.THRESH_BINARY)
            pre_image = np.reshape(pre_image, (image_shape, image_shape, 1))
            _, pre_img_png = cv2.imencode('.png', pre_image)
            pre_img_base64 = base64.b64encode(pre_img_png).decode()
            pre_images.append(pre_img_base64)
        else:
            pre_images.append(img_base64)
    else:
        dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "./dataset", project_name))
        x_train = np.load(os.path.join(dataset_dir, "x_train.npy"))
        y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
        image_shape = int(config['input_info']['change_shape'])
        preprocessing = config['input_info']['preprocessing']
        transform_list = [
            transforms.Resize((image_shape, image_shape)),
            transforms.ToTensor()
        ]
        if preprocessing == 'GCN':
            gcn = GCN()
            transform_list.append(gcn)
        elif preprocessing == 'ZCA':
            zca = ZCAWhitening.load(os.path.join(dataset_dir, f"{project_name}_zca.pth"))
            transform_list.append(zca)
        transform = transforms.Compose(transform_list)
        train_dataset = PreDataset(x_train, y_train, project_name, transform)
        train_loader = DataLoader(train_dataset, batch_size=int(1), shuffle=True)
        i = 0
        images = []
        pre_images = []
        label_list = []
        for x, pre, label in train_loader:
            label_list.append(label[0])
            numpy_x = x.numpy()
            numpy_x = np.transpose(numpy_x[0], (1, 2, 0))
            numpy_pre = pre.numpy()
            numpy_pre = np.transpose(numpy_pre[0], (1, 2, 0))
            print(x.shape)
            print(numpy_x.shape)

            _, img_png = cv2.imencode('.png', numpy_x*255)
            img_base64 = base64.b64encode(img_png).decode()
            images.append(img_base64)
            _, pre_img_png = cv2.imencode('.png', deprocess(numpy_pre)*255)
            pre_img_base64 = base64.b64encode(pre_img_png).decode()
            pre_images.append(pre_img_base64)
            i += 1
            if i == 10:
                break
    return images, pre_images, label_list


if __name__ == '__main__':
    config = {
        'project_name': 'CIFAR10'
    }
    get_images(config=config)