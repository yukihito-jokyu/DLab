import React, { useEffect, useState } from 'react';
import './EditTileParamet.css';
import { getOriginShape } from '../../db/firebaseFunction';

function EditTileParamet({ name, value, handleChangeParameter }) {
  // console.log(typeof(value))
  const [selectedValue, setSelectedValue] = useState('');
  const [floatValue, setFloatValue] = useState('');
  const [originShape, setOriginShape] = useState('');
  // モデルのオリジン画像サイズを取得
  useEffect(() => {
    const fatchOriginImageShape = async () => {
      const modelId = JSON.parse(sessionStorage.getItem('modelId'));
      const shape = await getOriginShape(modelId);
      // console.log(typeof(shape))
      // console.log(shape)
      setOriginShape(shape);
      // setOriginShape('5');
    };
    fatchOriginImageShape();
  }, []);
  useEffect(() => {
    const initValue = () => {
      if (typeof(value) === 'number') {
        if (value < 1 && value > 0) {
          setFloatValue(value);
        } else {
          setSelectedValue(value);
        }
      } else {
        setSelectedValue(value);
      }
    }
    initValue();
  }, [value]);
  const handleFloatChange = (e) => {
    const inputValue = e.target.value;
    const regex = /^[0-9.]*$/;
    if (regex.test(inputValue)) {
      // 入力が空か、有効な数値形式かチェック
      if (inputValue === '' || /^0(\.\d{0,3})?$/.test(inputValue)) {
        // 数値に変換して範囲をチェック
        const numValue = parseFloat(inputValue);
        if (isNaN(numValue) || (numValue >= 0 && numValue < 1)) {
          setFloatValue(inputValue);
        }
      }
    }
  };

  const handleChange = (e) => {
    setSelectedValue(e.target.value);
    handleChangeParameter(name, e.target.value);
  };
  return (
    <div className='edit-tile-paramet-wrapper'>
      <div className='edit-tile-name-wrapper'>
        <p>{name}</p>
      </div>
      <div className='edit-tile-value-wrapper'>
        {name === 'activ_func' ? (
          <select value={selectedValue} onChange={handleChange}>
            <option value="ReLU">ReLU</option>
            <option value="Sigmoid">Sigmoid</option>
            <option value="Tanh">Tanh</option>
            <option value="Softmax">Softmax</option>
          </select>
        ) : name === 'kernel_size' ? (
          <select value={selectedValue} onChange={handleChange}>
            {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        ) : name === 'out_channel' ? (
          <select value={selectedValue} onChange={handleChange}>
            {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        ) : name === 'padding' ? (
          <select value={selectedValue} onChange={handleChange}>
            {Array.from({ length: 200 }, (_, index) => index).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        ) : name === 'strid' ? (
          <select value={selectedValue} onChange={handleChange}>
            {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        ) : name === 'dropout_p' ? (
          <input
            type='text'
            value={floatValue}
            onChange={handleFloatChange}
            placeholder='0 以上 1 未満の数値'
          />
        ) : name === 'input_size' ? (
          <select value={selectedValue} onChange={handleChange}>
            {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        ) : name === 'preprocessing' ? (
          <select value={selectedValue} onChange={handleChange}>
            <option value="None">None</option>
            <option value="GCN">GCN</option>
            <option value="ZCA">ZCA</option>
          </select>
        ) : name === 'way' ? (
          <select value={selectedValue} onChange={handleChange}>
            <option value="normal">normal</option>
            <option value="GAP">GAP</option>
            <option value="GMP">GMP</option>
          </select>
        ) : name === 'changeShape' ? (
          <select value={selectedValue} onChange={handleChange}>
            {Array.from({ length: parseInt(originShape, 10) }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        ) : (
          <></>
        )}
      </div>
    </div>
  )
}

export default EditTileParamet
