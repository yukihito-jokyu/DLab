from train.train_cartpole import train_cartpole
from train.train_image_classification import train_model
from utils.db_manage import update_status, save_result

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
        update_status(model_id, 'doing')
        accuracy, loss = train_model(datas)
        save_result(model_id, accuracy, loss)
        update_status(model_id, 'done')
        socketio.emit('image_train_end'+model_id, {'message': 'train end'})