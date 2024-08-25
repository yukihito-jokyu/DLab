import React, { useEffect, useState } from 'react';
import './EditTileParamet.css';
import { getOriginShape } from '../../db/function/model_structure';
import { useParams } from 'react-router-dom';

function EditTileParamet({ name, value, handleChangeParameter }) {
  const [selectedValue, setSelectedValue] = useState('');
  const [floatValue, setFloatValue] = useState('');
  const [originShape, setOriginShape] = useState('');
  const { modelId, projectName } = useParams();
  // モデルのオリジン画像サイズを取得
  useEffect(() => {
    const fatchOriginImageShape = async () => {
      
      const shape = await getOriginShape(modelId);
      setOriginShape(shape);
    };
    fatchOriginImageShape();
  }, [modelId]);
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
  const paramName = {
    'change_shape': '入力サイズ',
    'preprocessing': '前処理',
    'activation_function': '活性化関数',
    'kernel_size': 'カーネルサイズ',
    'out_channel': '出力チャンネル数',
    'padding': 'パディング',
    'strid': 'ストライド',
    'dropout_p': 'ドロップアウト率',
    'way': 'ベクトル化手法',
    'neuron_size': 'ニューロン数'
  }
  return (
    <div className='edit-tile-paramet-wrapper'>
      <div className='edit-tile-name-wrapper'>
        <p>{paramName[name]}</p>
      </div>
      <div className='edit-tile-value-wrapper'>
        {name === 'activation_function' ? (
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
        ) : name === 'neuron_size' ? (
          <select value={selectedValue} onChange={handleChange}>
            {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
              <option key={number} value={number}>{number}</option>
            ))}
          </select>
        ) : name === 'preprocessing' ? (
          <select value={selectedValue} onChange={handleChange}>
            <option value="None">None</option>
            {projectName === "FlappyBird" || projectName === "CartPole" ? (
              <option value="GRAY">GRAY</option>
            ) : (
              <>
                <option value="GCN">GCN</option>
                <option value="ZCA">ZCA</option>
              </>
            )}
          </select>
        ) : name === 'way' ? (
          <select value={selectedValue} onChange={handleChange}>
            <option value="normal">normal</option>
            <option value="GAP">GAP</option>
            <option value="GMP">GMP</option>
          </select>
        ) : name === 'change_shape' ? (
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
