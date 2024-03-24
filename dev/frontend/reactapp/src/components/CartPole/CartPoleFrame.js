import React, { useContext, useState } from 'react';
import './CartPoleFrame.css';
import stick from '../../img/CartPole/stick.png'
import stand from '../../img/CartPole/stand.png'
import SocketContext from '../..';
import RadianToDegree from '../../utils/RadianToDegree';
import ToPx from '../../utils/ToPx';
import Getdata from '../../utils/Getdata';
import { InputContext } from '../../page/Cartpole';
import FileUpload from '../utils/FileUpload';

function CartPoleFrame(props) {
  // 引数の取得
  const id = props.id;
  // データの取得
  const { structures } = useContext(InputContext);
  // ソケット通信用
  const socket = useContext(SocketContext);
  // 回転
  const [rotation, setRotation] = useState(0);
  // 横移動
  const [move, setMove] = useState(0);
  // エピソード数
  const [episode, setEpisode] = useState(0);
  // transition
  const [disableTransition, setDisableTransition] = useState(false);
  // ソケット通信
  const handleTrain = () => {
    const AllData = Getdata(structures);
    const SentData = {
      CartPole: AllData,
      id: id
    };
    socket.emit('CartPole', SentData);
  };
  socket.on('CartPole_data'+id, (data) => {
    const newrotation = RadianToDegree(data.radian);
    const newmove = ToPx(data.location);
    setDisableTransition(false);
    setRotation(newrotation);
    setMove(newmove);
  });
  socket.on('end_CartPole', (data) => {
    setRotation(0);
    setMove(0);
  });
  socket.on('episode_start'+String(id), (data) => {
    const newrotation = RadianToDegree(data.radian);
    const newmove = ToPx(data.location);
    const newepisode = data.episode;
    setDisableTransition(true);
    setRotation(newrotation);
    setMove(newmove);
    setEpisode(newepisode);
  });
  socket.on('end', (data) => {
    console.log(data.message);
    setDisableTransition(true);
    setRotation(0);
    setMove(0);
  });
  // スタイル
  const QulleyStyle = {
    transition: disableTransition ? 'none' : 'transform 0.3s ease',
    transform: `translateX(${move}px)`,
  };
  const Rotatable = {
    transition: disableTransition ? 'none' : 'transform 0.3s ease',
    transform: `rotate(${rotation}deg)`
  };

  // データの取得
  const handleMakeConfig = async () => {
    const AllData = Getdata(structures);
    const response = await fetch('http://127.0.0.1:5000/CartPole/make/config', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(AllData),
    });
    const result = await response.json();
    console.log(result);
  };
  const handleTry = () => {
    socket.emit('CartPoleTry', {});
  };

  // pth,pyファイルの読み込み
  const handlePthUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('http://127.0.0.1:5000/CartPole/pth/upload', {
      method: 'POST',
      body: formData,
    });
    const result = await response.json();
    console.log(result);
  };
  const handlePyUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('http://127.0.0.1:5000/CartPole/py/upload', {
      method: 'POST',
      body: formData,
    });
    const result = await response.json();
    console.log(result);
  };

  return (
    <div>
      <button onClick={handleMakeConfig}>Config作成</button>
      <button onClick={handleTrain}>学習開始</button>
      <FileUpload onFileChange={handlePthUpload} />
      <FileUpload onFileChange={handlePyUpload} />
      <button onClick={handleTry}>お試し</button>
      <h1>CartPoleFrame</h1>
      <div>Episode:{episode}</div>
      <div className='frame'>
        <div className='pulley' style={QulleyStyle}>
          <div className='stick'>
            <img 
              className='rotatable'
              src={stick}
              alt='stick'
              style={Rotatable}
            />
          </div>
          <div className='stand'>
            <img 
              className='movetable'
              src={stand}
              alt='stand'
            />
          </div>
        </div>
        <div className='field'></div>
      </div>
    </div>
  );
};

export default CartPoleFrame;
