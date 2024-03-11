import React, { useState, useEffect } from 'react';

import './FlappyConvStyle.css';

function FlappyConvStyle(props) {
  const [isConv, setIsConv] = useState(true);
  const [isPool, setIsPool] = useState(false);
  const [nuronType, setNuronType] = useState('Conv2d')

  // 畳み込み層用
  const [inChannel, setInChannel] = useState(3);
  const [outChannel, setOutChannel] = useState(64);
  const [selectedActivation, setSelectedActivation] = useState('ReLU');
  // 共有
  const [kernel, setKernel] = useState(3);
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);

  // アウトプットサイズ
  const [outputH, setOutputH] = useState(0);
  const [outputW, setOutputW] = useState(0);

  // 引数取得
  const { DeletIvent, Index } = props

  // 出力のサイズを計算する
  useEffect(() => {
    let inputShape = null;
    if (Index === 0) {
      inputShape = document.querySelector('.Input_size').textContent.match(/\d+/g); 
    } else {
      inputShape = document.getElementById('Conv-structure').querySelector('.nuron-data').querySelector(`[class="${Index - 1}"]`).textContent.match(/\d+/g);
    }
    console.log('起動');
    console.log(inputShape);
    const H = ((Number(inputShape[0])+padding*2-kernel)/stride-1);
    const W = ((Number(inputShape[1])+padding*2-kernel)/stride-1);
    console.log(H, W);
    setOutputH(H);
    setOutputW(W);
    console.log(inputShape[0]);
  }, [outChannel, kernel, stride, padding, Index]);

  // チェックボックスの変化
  const handleCheckbox1Change = () => {
    // checkbox1がfalseだった時checkbox1をtureにしcheckbox2をfalseにする
    if (!isConv) {
      setIsConv(true);
      setIsPool(false);
      setNuronType('Conv2d');
    }
  };
  const handleCheckbox2Change = () => {
    // checkbox2がfalseだった時checkbox2をtureにしcheckbox1をfalseにする
    if (!isPool) {
      setIsConv(false);
      setIsPool(true);
      setNuronType('MaxPool2d');
    }
  };

  const handleInChannelChange = (e) => {
    setInChannel(parseInt(e.target.value, 10));
  };

  const handleOutChannelChange = (e) => {
    setOutChannel(parseInt(e.target.value, 10));
  };

  const handleKernelChange = (e) => {
    setKernel(parseInt(e.target.value, 10));
  };

  const handleStrideChange = (e) => {
    setStride(parseInt(e.target.value, 10));
  };

  const handlePaddingChange = (e) => {
    setPadding(parseInt(e.target.value, 10));
  };

  const handleActivationChange = (e) => {
    setSelectedActivation(e.target.value);
  };

  // クリックされたら要素を消す
  const handledelet = () => {
    DeletIvent();
  };
  return {
    value: Index,
    element: (
    <div className='conv-wrapper'>
      <div>
        <button onClick={handledelet}>-</button>
      </div>
      <div>
        <label>
          <input type='checkbox' checked={isConv} onChange={handleCheckbox1Change} />
          Conv2d
        </label>
        <label>
          <input type='checkbox' checked={isPool} onChange={handleCheckbox2Change} />
          MaxPool2d
        </label>
        <p className='Layer_name'>{nuronType}</p>
      </div>
      {/* 畳み込み層のhtml */}
      {isConv && <div>
        <label>入力データのチャンネル数：</label>
        <select value={inChannel} onChange={handleInChannelChange}>
          {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='In_channel'>{inChannel}</p>
      </div>}
      {isConv && <div>
        <label>出力データのチャンネル数：</label>
        <select value={outChannel} onChange={handleOutChannelChange}>
          {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='Out_channel'>{outChannel}</p>
      </div>}
      {isConv && <div>
        <label>畳み込みフィルタのサイズ：</label>
        <select value={kernel} onChange={handleKernelChange}>
          {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='Kernel_size'>{kernel}</p>
      </div>}
      {isConv && <div>
        <label>畳み込みフィルタの移動幅：</label>
        <select value={stride} onChange={handleStrideChange}>
          {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='Stride'>{stride}</p>
      </div>}
      {isConv && <div>
        <label>入力の周囲に追加されるパディング数：</label>
        <select value={padding} onChange={handlePaddingChange}>
          {Array.from({ length: 200 }, (_, index) => index).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='Padding'>{padding}</p>
      </div>}
      {isConv && <div className='activ-func'>
        <label htmlFor="activFuncSelector">活性化関数：</label>
        <select id="activFuncSelector" value={selectedActivation} onChange={handleActivationChange}>
          <option value="ReLU">ReLU</option>
          <option value="Sigmoid">Sigmoid</option>
          <option value="Tanh">Tanh</option>
          <option value="Softmax">Softmax</option>
          <option value="None">None</option>
        </select>
        <p className='Active_func'>{selectedActivation}</p>
      </div>}
      {/* MaxPoolのhtml */}
      {isPool && <div>
        <label>プーリングフィルタのサイズ：</label>
        <select value={kernel} onChange={handleKernelChange}>
          {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='Kernel_size'>{kernel}</p>
      </div>}
      {isPool && <div>
        <label>プーリングフィルタの移動幅：</label>
        <select value={stride} onChange={handleStrideChange}>
          {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='Stride'>{stride}</p>
      </div>}
      {isPool && <div>
        <label>入力の周囲に追加されるパディング数：</label>
        <select value={padding} onChange={handlePaddingChange}>
          {Array.from({ length: 200 }, (_, index) => index).map((number) => (
            <option key={number} value={number}>{number}</option>
          ))}
        </select>
        <p className='Padding'>{padding}</p>
      </div>}
      <div>
        <p className={Index}>アウトプットサイズ：({outputH}, {outputW}, {outChannel})</p>
      </div>
    </div>
  )};
}

export default FlappyConvStyle;
