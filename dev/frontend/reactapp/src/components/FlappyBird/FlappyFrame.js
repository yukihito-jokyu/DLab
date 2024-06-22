import React, { useState, useContext } from 'react';
// import SocketContext from '../..';
import GetFlappyData from '../../utils/GetFlappyData';
import { CNNContext } from '../../page/Flappybird';

function FlappyFrame(props) {
  const id = props.id;
  const [imagedata, setImagedata] = useState(null);
  // const socket = useContext(SocketContext);
  const { trainInfo } = useContext(CNNContext);

  // const handleTrainFlappy = () => {
  //   const alldata = GetFlappyData(trainInfo);
  //   console.log(alldata);
  //   const sendData = {
  //     alldata: alldata,
  //     id: id
  //   }
  //   socket.emit('FlappyBird', sendData);
  // };

  // socket.on('FlappyBird_data'+id, (data) => {
  //   const newimagedata = data.image_data;
  //   setImagedata(newimagedata);
  // });

  const handleGetData = () => {
    const alldata = GetFlappyData(trainInfo);
    console.log(alldata);
  }


  return (
    <div>
      <button onClick={handleGetData}>データ取得</button>
      {/* <button onClick={handleTrainFlappy}>学習開始</button> */}
      <div>
        {imagedata && <img src={`data:image/png;base64,${imagedata}`} alt="test_Image" />}
      </div>
    </div>
  );
}

export default FlappyFrame;
