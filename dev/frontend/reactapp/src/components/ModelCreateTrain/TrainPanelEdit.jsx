import React, { useEffect, useState } from 'react';
import './TrainPanelEdit.css';
import { ReactComponent as InfoIcon } from '../../assets/svg/info_24.svg';

function TrainPanelEdit({ parameter, value, handleChangeParameter, setInformation, setParamName }) {
  const [selectedValue, setSelectedValue] = useState('');
  const [inputValue, setInputValue] = useState(value);

  useEffect(() => {
    setSelectedValue(value);
    setInputValue(value);
  }, [value]);

  const handleNumChange = (e, min, max) => {
    const val = e.target.value;
    const num = parseInt(val, 10);
    if (val === '' || (Number.isInteger(num) && num >= min && num <= max)) {
      setInputValue(val);
      handleChangeParameter(parameter, val === '' ? '' : num);
    }
  };

  const handleFloatChange = (e, min, max) => {
    const val = e.target.value;
    const num = parseFloat(val);
    if (val === '' || (!isNaN(num) && num >= min && num <= max)) {
      setInputValue(val);
      handleChangeParameter(parameter, val === '' ? '' : num);
    }
  };

  const handleChange = (e) => {
    setSelectedValue(e.target.value);
    handleChangeParameter(parameter, e.target.value);
  };

  const handleBooleanSelectChange = (e) => {
    const val = e.target.value === 'true';
    handleChangeParameter(parameter, val);
  };

  const handleInfoButton = () => {
    setInformation(true);
    setParamName(paramName[parameter]);
  };

  const paramName = {
    // 学習のパラメータ
    'batch': 'バッチサイズ',
    'epoch': '学習回数',
    'learning_rate': '学習率',
    'optimizer': '最適化手法',
    'test_size': '検証データの割合',
    'buffer': '経験再生バッファ数',
    'episilon': '探索率',
    'syns': '同期間隔',
    // データ拡張
    'rotation_degrees': '回転角度',
    'vertical_translation_factor': '上下平行移動係数',
    'horizontal_translation_factor': '左右平行移動係数',
    'scaling_factor': 'スケーリング係数',
    'zoom_factor': 'ズーム係数',
    'brightness_factor': '明るさ係数',
    'contrast_factor': 'コントラスト係数',
    'saturation_factor': '彩度係数',
    'hue_factor': '色相係数',
    'sharpness_factor': 'シャープネス係数',
    'shear_angle': 'せん断角度',
    'noise_factor': 'ノイズ倍率',
    'do_flipping': '水平反転',
    'do_vertical_flipping': '垂直反転',
    'grayscale_p': 'グレースケール適用確率'
  };

  return (
    <>
      {parameter !== 'image_shape' ? (
        <div className='train-panel-edit-wrapper'>
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
            ) : parameter === 'batch' || parameter === 'epoch' || parameter === 'syns' ? (
              <select value={selectedValue} onChange={handleChange}>
                {Array.from({ length: 200 }, (_, index) => index + 1).map((number) => (
                  <option key={number} value={number}>{number}</option>
                ))}
              </select>
            ) : parameter === 'optimizer' ? (
              <select value={selectedValue} onChange={handleChange}>
                <option value="SGD">SGD</option>
                <option value="momentum">momentum</option>
                <option value="Adam">Adam</option>
                <option value="Adagrad">Adagrad</option>
                <option value="RMSprop">RMSprop</option>
                <option value="Adadelta">Adadelta</option>
              </select>
            ) : parameter === 'learning_rate' || parameter === 'episilon' || parameter === 'test_size' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleFloatChange(e, 0, 1)}
                step='0.0001'
                min='0'
                max='1'
              />
            ) : parameter === 'buffer' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleNumChange(e, 0, 100000)}
                min='0'
                max='100000'
              />
            ) : parameter === 'rotation_degrees' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleNumChange(e, 0, 360)}
                min='0'
                max='360'
              />
            ) : parameter === 'vertical_translation_factor' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleFloatChange(e, 0, 1)}
                step='0.01'
                min='0'
                max='1'
              />
            ) : parameter === 'horizontal_translation_factor' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleFloatChange(e, 0, 1)}
                step='0.01'
                min='0'
                max='1'
              />
            ) : parameter === 'scaling_factor' || parameter === 'zoom_factor' || parameter === 'sharpness_factor' ||
              parameter === 'contrast_factor' || parameter === 'brightness_factor' || parameter === 'saturation_factor' ||
              parameter === 'hue_factor' || parameter === 'cutout_scale_factor' || parameter === 'noise_factor' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleFloatChange(e, 0, 1)}
                step='0.01'
                min='0'
                max='1'
              />
            ) : parameter === 'cutout_p' || parameter === 'grayscale_p' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleFloatChange(e, 0, 1)}
                step='0.01'
                min='0'
                max='1'
              />
            ) : parameter === 'cutout_aspect_ratio' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleFloatChange(e, 0.1, 10)}
                step='0.1'
                min='0.1'
                max='10'
              />
            ) : parameter === 'do_flipping' || parameter === 'do_vertical_flipping' ? (
              <select value={value.toString()} onChange={handleBooleanSelectChange}>
                <option value="true">True</option>
                <option value="false">False</option>
              </select>
            ) : (
              <input
                type='text'
                value={inputValue}
                onChange={handleChange}
              />
            )}
          </div>
        </div>
      ) : null}
    </>
  );
}

export default TrainPanelEdit;
