from train.train_cartpole import train_cartpole
from train.train_image_classification import train_model
from train.train_flappybird import train_flappy
from utils.db_manage import update_status, save_result_manegement, save_result_readarboard

def setup_sockets(socketio):
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
    
    @socketio.on('test')
    def test(data):
        print('test data:', data)
        for i in range(10):
            socketio.sleep(1)
            socketio.emit('test_event', {'response': i})
            print(i)
    
    @socketio.on('cartpole_train_start')
    def train_CartPole(datas):
        socketio.sleep(0.5)
        print(datas)
        model_id = datas['model_id']
        project_name = datas["project_name"]
        user_id = datas["user_id"] 
        user_name = datas["user_name"]
        update_status(model_id, 'doing')
        total_reward, loss = train_cartpole(datas, socketio)
        save_result_manegement(model_id, total_reward, loss)
        save_result_readarboard(project_name, user_id, user_name, total_reward)
        update_status(model_id, 'done')
        socketio.emit('cartpole_train_end'+model_id, {'message': 'train end'})
    
    @socketio.on('flappy_train_start')
    def flappy_train(datas):
        socketio.sleep(0.5)
        print(f"\ndatas:{datas}\n")
        model_id = datas['model_id']
        project_name = datas["project_name"]
        user_id = datas["user_id"] 
        user_name = datas["user_name"]
        update_status(model_id, 'doing')
        total_reward, loss = train_flappy(datas, socketio)
        save_result_manegement(model_id, total_reward, loss)
        save_result_readarboard(project_name, user_id, user_name, total_reward)
        update_status(model_id, 'done')
        socketio.emit('flappy_train_end'+model_id, {'message': 'train end'})

    @socketio.on('image_train_start')
    def train(datas):
        socketio.sleep(0.5)
        print(datas)
        model_id = datas['model_id']
        project_name = datas["project_name"]
        user_id = datas["user_id"]
        print(f"\ndatas:{datas}\n")
        user_name = datas["user_name"]
        update_status(model_id, 'doing')
        accuracy, loss = train_model(datas, socketio)
        save_result_manegement(model_id, accuracy, loss)
        save_result_readarboard(project_name, user_id, user_name, accuracy)
        update_status(model_id, 'done')
        socketio.emit('image_train_end'+model_id, {'message': 'train end'})