def setup_sockets(socketio):
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
    
    @socketio.on('test')
    def test(data):
        print('test data:', data)
        socketio.emit('test_event', {'response': 'Data received'})