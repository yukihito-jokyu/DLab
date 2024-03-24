import React, { useContext, useEffect, useState } from 'react'
import { CNNContext } from '../../page/ImageRecognition'

function InputLayer() {
  // Contextの読み込み
  const { trainInfo } = useContext(CNNContext);
  const [inputSize, setInputSize] = trainInfo.inputLayer;
  // useState
  const [checkbox1, setCheckbox1] = useState(true);
  const [checkbox2, setCheckbox2] = useState(false);
  const [c, setC] = useState(1);
  const [h, setH] = useState(28);
  const [w, setW] = useState(28);

  // チェックボックスの変化
  const handleCheckbox1Change = () => {
    // checkbox1がfalseだった時checkbox1をtureにしcheckbox2をfalseにする
    if (!checkbox1) {
      setCheckbox1(true);
      setCheckbox2(false);
      setC(1);
    }
  };
  const handleCheckbox2Change = () => {
    // checkbox2がfalseだった時checkbox2をtureにしcheckbox1をfalseにする
    if (!checkbox2) {
      setCheckbox1(false);
      setCheckbox2(true);
      setC(3);
    }
  };

  // 高さ幅の変化
  const handleHChange = (e) => {
    setH(parseInt(e.target.value, 10));
  };
  const handleWChange = (e) => {
    setW(parseInt(e.target.value, 10));
  };

  // 特定の変数が変化したら実行する
  useEffect(() => {
    setInputSize([h, w, c]);
  }, [h, w, c, setInputSize]);

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
          <p>元データサイズ：(28, 28, 1) (H×W×C)</p>
        </div>
        <div>
          <label htmlFor='H'>高さ：</label>
          <select id='H' value={h} onChange={handleHChange}>
            {Array.from({ length: 300 }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor='W'>幅：</label>
          <select id='W' value={w} onChange={handleWChange}>
            {Array.from({ length: 300 }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        </div>
        <div>
          <p className='Input_size'>前処理後のデータサイズ：(
            {inputSize.map((value, index) => (
              <React.Fragment key={index}>
                <span>{value}</span>
                {index !== inputSize.length - 1 && <span>, </span>}
              </React.Fragment>
              ))}
            )
          </p>
        </div>
      </div>
    </div>
  )
}

export default InputLayer
