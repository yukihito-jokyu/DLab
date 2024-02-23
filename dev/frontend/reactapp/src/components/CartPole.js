import React, { useContext, useState } from 'react';
import './CartPole.css';
import stick from '../img/CartPole/stick.png'
import stand from '../img/CartPole/stand.png'
import SocketContext from '..';
import RadianToDegree from '../utils/RadianToDegree';
import ToPx from '../utils/ToPx';

function CartPole() {
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
  // test
  const handleRotate = () => {
    setRotation(rotation + 45);
  };
  const handleMove = () => {
    setMove(move + 10);
  };
  // ソケット通信
  socket.on('CartPole_data', (data) => {
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
  socket.on('episode_start', (data) => {
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
  })
  const handleSocket = () => {
    socket.emit('test_CartPole', {});
  };
  // スタイル
  const QulleyStyle = {
    transition: disableTransition ? 'none' : 'transform 0.3s ease',
    transform: `translateX(${move}px)`,
  };
  const Rotatable = {
    transition: disableTransition ? 'none' : 'transform 0.3s ease',
    transform: `rotate(${rotation}deg)`
  }
  return (
    <div>
      <button onClick={handleSocket}>Socket通信</button>
      <button onClick={handleRotate}>回転</button>
      <button onClick={handleMove}>移動</button>
      <h1>CartPole</h1>
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

export default CartPole;
