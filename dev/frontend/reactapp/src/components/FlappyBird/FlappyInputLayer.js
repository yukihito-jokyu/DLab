import React, { useContext, useState } from 'react'
import SocketContext from '../..';

import FlappyImage from '../../assets/images/FlappyBird.png'

import './FlappyInputLayer.css'

function FlappyInputLayer() {
  const [checkbox1, setCheckbox1] = useState(true);
  const [checkbox2, setCheckbox2] = useState(false);
  const [channel, setChannel] = useState(1);

  const [selecth, setSelecth] = useState(160);
  const [selectw, setSelectw] = useState(160);

  const [imagedata, setImagedata] = useState(null);

  // ソケット通信用
  const socket = useContext(SocketContext);

  // チェックボックスの変化
  const handleCheckbox1Change = () => {
    // checkbox1がfalseだった時checkbox1をtureにしcheckbox2をfalseにする
    if (!checkbox1) {
      setCheckbox1(true);
      setCheckbox2(false);
      setChannel(1);
    }
  };
  const handleCheckbox2Change = () => {
    // checkbox2がfalseだった時checkbox2をtureにしcheckbox1をfalseにする
    if (!checkbox2) {
      setCheckbox1(false);
      setCheckbox2(true);
      setChannel(3);
    }
  };

  // 高さ幅の変化
  const handleHChange = (e) => {
    setSelecth(parseInt(e.target.value, 10));
  };
  const handleWChange = (e) => {
    setSelectw(parseInt(e.target.value, 10));
  };

  // socket通信
  socket.on('get_resize_image', (data) => {
    const newimagedata = data.image_data;
    setImagedata(newimagedata);
  });

  //クリックイベント
  const handleSentSize = () => {
    const ImageSize = {
      'H': selecth,
      'W': selectw,
      'C': channel
    };
    // socket通信開始
    socket.emit('InputImage', ImageSize);
  };

  return (
    <div>
      <h1>入力画像の処理</h1>
      <div className='Flappy-Input-Wrapper'>
        <div className='check-box'>
          <label>
            <input type='checkbox' checked={checkbox1} onChange={handleCheckbox1Change} />
            白黒
          </label>
          <label>
            <input type='checkbox' checked={checkbox2} onChange={handleCheckbox2Change} />
            カラー
          </label>
        </div>
        <div>
          <p>元データサイズ：(512, 288, 3) (H×W×C)</p>
        </div>
        <div>
          <label htmlFor='H'>高さ：</label>
          <select id='H' value={selecth} onChange={handleHChange}>
            {Array.from({ length: 300 }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor='W'>幅：</label>
          <select id='W' value={selectw} onChange={handleWChange}>
            {Array.from({ length: 300 }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        </div>
        <div>
          <p className='Input_size'>前処理後のデータサイズ：({selecth}, {selectw}, {channel})</p>
        </div>
        <div>
          <button onClick={handleSentSize}>確認</button>
        </div>
      </div>
      <div className='flappy-images'>
        <div>
          <p>処理前</p>
          <img src={FlappyImage} alt='source' />
        </div>
        <div><p>➡</p></div>
        <div>
          <p>処理後</p>
          {imagedata && <img className='pre-image' src={`data:image/png;base64,${imagedata}`} alt="pre_Image" />}
        </div>
      </div>
    </div>
  )
}

export default FlappyInputLayer
