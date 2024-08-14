import numpy as np
import json

import cv2
import base64

import os
from torchvision import transforms
from train.train_image_classification import CustomDataset, ZCAWhitening, GCN

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


# 前処理後の画像の取得
def get_images(project_name, config):
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "./dataset", project_name))
    x_train = np.load(os.path.join(dataset_dir, "x_train.npy"))
    y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
    image_shape = int(config['change_shape'])
    transform_list = [
        transforms.Resize((image_shape, image_shape)),
        transforms.ToTensor()
    ]
    # if config['propres']