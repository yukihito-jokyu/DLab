import React, { useState, useContext } from 'react'
import SocketContext from '..';

function FlappyBird() {
  const [imagedata, setImagedata] = useState(null);
  const socket = useContext(SocketContext);

  // socket.on('FlappyBird_data', (data) => {
  //   const newimagedata = data.image_data;
  //   setImagedata(newimagedata);
  // });

  // const handleFlappy = () => {
  //   socket.emit('FlappyBird', {})
  // }


  return (
    <div>
      {/* <button onClick={handleFlappy}>FlappyBird</button> */}
      <div className='Flappy_frame'>
        {imagedata && <img src={`data:image/png;base64,${imagedata}`} alt="test_Image" />}
      </div>
    </div>
  )
}

export default FlappyBird
