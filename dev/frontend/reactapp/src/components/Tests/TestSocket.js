import React, { useContext, useState } from 'react'
import SocketContext from '../..';

function TestSocket(props) {
  const socket = useContext(SocketContext);
  const id = props.id;
  const [data, setData] = useState('');

  const handleSocket = () => {
    const sentData = {
      id: id
    };
    console.log(id)
    socket.emit('test_socket', sentData);
  }
  socket.on('socket_test', (data) => {
    console.log(data.data);
    setData(data.data);
  })
  return (
    <div
      style={{
        border: '0.5px solid #1c0909',
        width: '400px',
      }}
    >
      <p>{id}</p>
      <button onClick={handleSocket}>通信</button>
      <p>受信したデータ：{data}</p>
    </div>
  )
}

export default TestSocket;