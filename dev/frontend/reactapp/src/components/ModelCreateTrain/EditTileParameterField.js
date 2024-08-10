import React, { useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import EditTileParamet from './EditTileParamet';
import { v4 as uuidv4 } from 'uuid';
import InputLayer from '../Image/InputLayer';
import { ReactComponent as InfoIcon } from '../../assets/svg/info_24.svg';

function EditTileParameterField({ parameter, inputLayer, convLayer, flattenWay, middleLayer, layerType, param, selectedindex, setInputLayer, setConvLayer, setFlattenWay, setMiddleLayer, setParam, setInfoModal, setInfoName }) {
  const pList = ["kernel_size", "activation_function", "out_channel", "padding", "strid", "dropout_p", "neuron_size", "preprocessing", "way", "change_shape"]
  const [keys, setKeys] = useState([]);
  useEffect(() => {
    const handleSetParameter = () => {
      let layerParam = null;

      if (layerType === "Conv") {
        layerParam = convLayer[selectedindex];
      } else if (layerType === "Middle") {
        layerParam = middleLayer[selectedindex];
      } else if (layerType === 'Input') {
        layerParam = inputLayer;
      } else if (layerType === 'Flatten') {
        layerParam = flattenWay;
      }

      if (layerParam) {
        setParam(layerParam);
        const keys = Object.keys(layerParam);
        const sortedKeys = keys.sort((a, b) => a.localeCompare(b));
        setKeys(sortedKeys);
      } else {
        setParam(null);
        setKeys([]);
      }
    }

    handleSetParameter();
  }, [parameter, inputLayer, layerType, selectedindex, setParam, convLayer, flattenWay, middleLayer]);

  const handleChangeParameter = (key, value) => {
    console.log(key, value)
    if (layerType === 'Conv') {
      const newConvLayer = [...convLayer];
      newConvLayer[selectedindex] = { ...newConvLayer[selectedindex], [key]: value }
      setConvLayer(newConvLayer);
    } else if (layerType === 'Middle') {
      const newMiddleLayer = [...middleLayer];
      newMiddleLayer[selectedindex] = { ...newMiddleLayer[selectedindex], [key]: value }
      setMiddleLayer(newMiddleLayer);
    } else if (layerType === 'Input') {
      const newInputLayer = { ...inputLayer };
      newInputLayer[key] = value;
      // console.log(newInputLayer['shape'][0])
      if (key === 'change_shape') {
        newInputLayer['shape'][0] = Number(value)
        newInputLayer['shape'][1] = Number(value)
      }
      setInputLayer(newInputLayer);
    } else if (layerType === 'Flatten') {
      const newFlattenWay = { ...flattenWay };
      newFlattenWay[key] = value;
      setFlattenWay(newFlattenWay);
    }
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
  const handleInfoModal = (key) => {
    setInfoModal(true)
    setInfoName(paramName[key])
  }
  return (
    <div className='edit-tile-parameter-wrapper'>
      <div className='edit-tile-field'>

        {param && keys.map((key, index) => (
          pList.includes(String(key)) && (
            <div key={index} className='edit-wrapper'>
              <div className='edit-field'>
                <div className='param-information-button'>
                  <div className='button-wrapper' onClick={() => handleInfoModal(key)} style={{ cursor: 'pointer' }}>
                    <InfoIcon className='info-icon' />
                  </div>
                </div>
                <EditTileParamet name={key} value={param[key]} handleChangeParameter={handleChangeParameter} />
              </div>
            </div>
          )
        ))}
      </div>
    </div>
  )
}

export default EditTileParameterField
