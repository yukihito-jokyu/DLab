import React, { useEffect, useState } from 'react';
import './TrainPanelEdit.css';
import { ReactComponent as InfoIcon } from '../../assets/svg/info_24.svg';

function TrainPanelEdit({ parameter, value, handleChangeParameter, setInformation, setParamName }) {
  const [selectedValue, setSelectedValue] = useState('');
  const [floatValue, setFloatValue] = useState('');
  const [testSize, setTestSize] = useState('');
  const [numValue, setNumValue] = useState('');
  useEffect(() => {
    setSelectedValue(value);
  }, [value]);
  const handleNumChange = (e) => {
    const inputValue = e.target.value;
    const regex = /^[0-9]*$/; // 数字のみを許可

    if (regex.test(inputValue)) {
      // 空入力を許可
      if (inputValue === '') {
        setNumValue('');
      } else {
        const num = parseInt(inputValue, 10);
        // 0より大きく100000以下の整数のみを許可
        if (num > 0 && num <= 100000) {
          setNumValue(inputValue);
        }
      }
    }
  };
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
    setSelectedValue(e.target.value);
    handleChangeParameter(parameter, e.target.value);
  };
  // test_size
  const handleTestSizeChange = (e) => {
    const inputValue = e.target.value;
    const regex = /^[0-9.]*$/;
    if (regex.test(inputValue)) {
      // 入力が空か、有効な数値形式かチェック
      if (inputValue === '' || /^0(\.\d{0,1})?$/.test(inputValue)) {
        // 数値に変換して範囲をチェック
        const numValue = parseFloat(inputValue);
        if (isNaN(numValue) || (numValue >= 0 && numValue < 1)) {
          setTestSize(inputValue);
        }
      }
    }
    setSelectedValue(e.target.value);
    handleChangeParameter(parameter, e.target.value);
  };
  const handleChange = (e) => {
    setSelectedValue(e.target.value);
    handleChangeParameter(parameter, e.target.value);
  };
  const handleInfoButton = () => {
    setInformation(true)
    setParamName(paramName[parameter])
  }
  const paramName = {
    'batch': 'バッチサイズ',
    'epoch': '学習回数',
    'learning_rate': '学習率',
    'optimizer': '最適化手法',
    'test_size': '検証データの割合'
  }
  return (
    <>
      {parameter !== 'image_shape' ? (<div className='train-panel-edit-wrapper'>
        <div className='param-information-button'>
          <div className='button-wrapper' onClick={handleInfoButton} style={{ cursor: 'pointer' }}>
            <InfoIcon className='info-icon' />
          </div>
        </div>
        <div className='train-panel-name-wrapper'>
          <p>{paramName[parameter]}</p>
        </div>
        <div className='train-panel-value-wrapper'>
          {parameter === 'loss' ? (
            <select value={selectedValue} onChange={handleChange}>
              <option value="mse_loss">mse_loss</option>
              <option value="cross_entropy">cross_entropy</option>
              <option value="binary_corss_entropy">binary_corss_entropy</option>
              <option value="nll_loss">nll_loss</option>
              <option value="hinge_embedding_loss">hinge_embedding_loss</option>
            </select>
          ) : parameter === 'batch' ? (
            <select value={selectedValue} onChange={handleChange}>
              {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
                <option key={number} value={number}>{number}</option>
              ))}
            </select>
          ) : parameter === 'epoch' ? (
            <select value={selectedValue} onChange={handleChange}>
              {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
                <option key={number} value={number}>{number}</option>
              ))}
            </select>
          ) : parameter === 'learning_rate' ? (
            <input
              type='text'
              value={floatValue}
              onChange={handleFloatChange}
              placeholder='0 以上 1 未満の数値'
            />
          ) : parameter === 'optimizer' ? (
            <select value={selectedValue} onChange={handleChange}>
              <option value="SGD">SGD</option>
              <option value="momentum">momentum</option>
              <option value="Adam">Adam</option>
              <option value="Adagrad">Adagrad</option>
              <option value="RMSprop">RMSprop</option>
              <option value="Adadelta">Adadelta</option>
            </select>
          ) : parameter === 'episilon' ? (
            <input
              type='text'
              value={floatValue}
              onChange={handleFloatChange}
              placeholder='0 以上 1 未満の数値'
            />
          ) : parameter === 'buffer' ? (
            <input
              type='text'
              value={numValue}
              onChange={handleNumChange}
              placeholder='1 以上 10万 以下の数値'
            />
          ) : parameter === 'syns' ? (
            <select value={selectedValue} onChange={handleChange}>
              {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
                <option key={number} value={number}>{number}</option>
              ))}
            </select>
          ) : parameter === 'test_size' ? (
            <input
              type='text'
              value={testSize}
              onChange={handleTestSizeChange}
              placeholder='0 以上 1 未満の数値'
            />
          ) : (
            <></>
          )}
        </div>
      </div>) : (
        <></>
      )}
    </>
  )
}

export default TrainPanelEdit
