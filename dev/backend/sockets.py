from train.train_cartpole import train_cartpole
from train.train_image_classification import train_model

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
        global thread
        train_cartpole(datas, socketio)

    @socketio.on('Train')
    def train(datas):
        global thread
        train_model(datas, socketio)