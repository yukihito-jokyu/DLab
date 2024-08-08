from train.train_cartpole import train_cartpole
from train.train_image_classification import train_model
from utils.db_manage import update_status, save_result_manegement, save_result_readarboard

def setup_sockets(socketio):
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
    
    @socketio.on('test')
    def test(data):
        print('test data:', data)
        socketio.emit('test_event', {'response': 'Data received'})
    
    @socketio.on('CartPole')
    def train_CartPole(datas):
        train_cartpole(datas, socketio)

    @socketio.on('image_train_start')
    def train(datas):
        print(datas)
        model_id = datas['model_id']
        project_name = datas["project_name"]
        user_id = datas["user_id"]
        print(f"\ndatas:{datas}\n")
        user_name = datas["user_name"]
        update_status(model_id, 'doing')
        accuracy, loss = train_model(datas)
        save_result_manegement(model_id, accuracy, loss)
        save_result_readarboard(project_name, user_id, user_name, accuracy)
        update_status(model_id, 'done')
        socketio.emit('image_train_end'+model_id, {'message': 'train end'})