from train.train_cartpole import DQNAgent
from utils.db_manage import upload_training_result, upload_file, initialize_training_results
from Flappy.test_flappy import GameState
import torch
from flask_socketio import emit
import tempfile
import matplotlib.pyplot as plt
import cv2
import matplotlib
import numpy as np
import base64
matplotlib.use('Agg')

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"FlappyBird:{device}")


def train_flappy(config):
    model_id = config["model_id"]
    user_id = config["user_id"]
    project_name = config["project_name"]
    train_info = config["Train_info"]
    image_shape = int(config['input_info']["change_shape"])
    epoch = int(train_info["epoch"])
    sync_interval = int(train_info["syns"])
    print(image_shape)
    print(type(image_shape))

    max_reward = 0
    rewards = []
    losses = []

    env = GameState()
    agent = DQNAgent(train_info=train_info, config=config, device=device)
    
    init_result = initialize_training_results(model_id, "ReinforcementLearning")
    print(init_result)

    for episode in range(1, int(epoch)+1):
        total_reward = 0
        total_loss = 0
        step = 0
        init_action = [0, 1]
        state, reward, done = env.frame_step(init_action)
        state = np.transpose(cv2.resize(state, (image_shape, image_shape)), (2, 0, 1))

        while not done:
            action = agent.get_action(state=state)
            action_list = np.zeros(2)
            action_list[action] = 1
            next_state, reward, done = env.frame_step(action_list)
            next_state = np.transpose(cv2.resize(next_state, (image_shape, image_shape)), (2, 0, 1))
            loss = agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            total_loss += loss

            if step == 200:
                break
            
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

        print(f'Epoch: {episode} TotalReward: {total_reward} AverageLoss: {round(ave_loss, 5)}')
        emit('train_flappy_results'+model_id, epoch_result)

        # 検証(10epochごと)
        if episode % 50 == 1:
            print(f'検証: {episode}')
            image_list = []
            with torch.no_grad():
                image, reward, done = env.frame_step(init_action)
                state = np.transpose(cv2.resize(image, (image_shape, image_shape)), (2, 0, 1))
                # 画像をバイナリデータへ変換
                _, origin_img_png = cv2.imencode('.png', image)
                img_base64 = base64.b64encode(origin_img_png).decode()
                image_list.append(img_base64)
                while not done:
                    action = agent.get_action(state=state)
                    action_list = np.zeros(2)
                    action_list[action] = 1
                    image, reward, done = env.frame_step(action_list)
                    state = np.transpose(cv2.resize(image, (image_shape, image_shape)), (2, 0, 1))
                    # 画像をバイナリデータへ変換
                    _, origin_img_png = cv2.imencode('.png', image)
                    img_base64 = base64.b64encode(origin_img_png).decode()
                    image_list.append(img_base64)
            # emitで画像を渡す
            images_data = {
                "Epoch": epoch,
                "Images": image_list
            }
            emit('flappy_valid'+str(model_id), images_data)
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