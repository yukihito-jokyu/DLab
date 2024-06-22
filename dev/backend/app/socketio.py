from __init__ import socketio

@socketio.on('message')
def handle_message(data):
    print('received message: ' + data)
    socketio.emit('response', {'data': 'Message received'})
