import React, { useState, useContext } from 'react';
import SocketContext from '../..';
import GetFlappyData from '../../utils/GetFlappyData';

function FlappyFrame() {
  const [imagedata, setImagedata] = useState(null);
  const socket = useContext(SocketContext);

  const handleTrainFlappy = () => {
    const alldata = GetFlappyData();
    console.log(alldata);
    socket.emit('FlappyBird', alldata);
  };

  socket.on('FlappyBird_data', (data) => {
    const newimagedata = data.image_data;
    setImagedata(newimagedata);
  });


  return (
    <div>
      <button onClick={handleTrainFlappy}>学習開始</button>
      <div>
        {imagedata && <img src={`data:image/png;base64,${imagedata}`} alt="test_Image" />}
      </div>
    </div>
  );
}

export default FlappyFrame;
