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
    setParamName(parameters[parameter]?.name || parameter);
  };

  const parameters = {
    // 学習のパラメータ
    'batch': { name: 'バッチサイズ', range: '1 〜 200' },
    'epoch': { name: '学習回数', range: '1 〜 200' },
    'learning_rate': { name: '学習率', range: '0 〜 1' },
    'optimizer': { name: '最適化手法', range: '選択肢に応じた手法' },
    'test_size': { name: '検証データの割合', range: '0 〜 1' },
    'buffer': { name: '経験再生バッファ数', range: '0 〜 100000' },
    'episilon': { name: '探索率', range: '0 〜 1' },
    'syns': { name: '同期間隔', range: '1 〜 200' },
    // データ拡張
    'rotation_degrees': { name: '回転角度', range: '0 〜 180' },
    'vertical_translation_factor': { name: '上下平行移動係数', range: '0 〜 1' },
    'horizontal_translation_factor': { name: '左右平行移動係数', range: '0 〜 1' },
    'scaling_factor': { name: 'スケーリング係数', range: '0 〜 1' },
    'zoom_factor': { name: 'ズーム係数', range: '0 〜 1' },
    'brightness_factor': { name: '明るさ係数', range: '0 〜 1' },
    'contrast_factor': { name: 'コントラスト係数', range: '0 〜 1' },
    'saturation_factor': { name: '彩度係数', range: '0 〜 1' },
    'hue_factor': { name: '色相係数', range: '0 〜 0.5' },
    'sharpness_factor': { name: 'シャープネス係数', range: '0 〜 1' },
    'shear_angle': { name: 'せん断角度', range: '0 〜 180' },
    'noise_factor': { name: 'ノイズ倍率', range: '0 〜 1' },
    'do_flipping': { name: '水平反転', range: 'True または False' },
    'do_vertical_flipping': { name: '垂直反転', range: 'True または False' },
    'grayscale_p': { name: 'グレースケール適用確率', range: '0 〜 1' }
  };

  return (
    <>
      {parameter !== 'image_shape' ? (
        <div className='train-panel-edit-wrapper'>
          <div className='param-information-button'>
            <div
              className='button-wrapper'
              onClick={handleInfoButton}
              style={{ cursor: 'pointer' }}
            >
              <InfoIcon className='info-icon' />
              <div className='tooltip'>
                {parameters[parameter]?.range || '範囲情報がありません'}
              </div>
            </div>
          </div>
          <div className='train-panel-name-wrapper'>
            <p>{parameters[parameter]?.name || parameter}</p>
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
                onChange={(e) => handleNumChange(e, 0, 180)}
                min='0'
                max='180'
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
              parameter === 'cutout_scale_factor' || parameter === 'noise_factor' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleFloatChange(e, 0, 1)}
                step='0.01'
                min='0'
                max='1'
              />
            ) : parameter === 'hue_factor' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleFloatChange(e, 0, 0.5)}
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
            ) : parameter === 'shear_angle' ? (
              <input
                type='number'
                value={inputValue}
                onChange={(e) => handleNumChange(e, 0, 180)}
                step='1'
                min='0'
                max='180'
              />
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