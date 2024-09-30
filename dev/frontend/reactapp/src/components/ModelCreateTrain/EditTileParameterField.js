import React, { useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import EditTileParamet from './EditTileParamet';
import { ReactComponent as InfoIcon } from '../../assets/svg/info_24.svg';
import { getOriginShape } from '../../db/function/model_structure';
import { useParams } from 'react-router-dom';
import TrainPanelTital from './TrainPanelTital';
import TrainPanelEdit from './TrainPanelEdit';
import InformationModal from './Modal/InformationModal';

function EditTileParameterField({ parameter, inputLayer, convLayer, flattenWay, middleLayer, layerType, param, selectedindex, setInputLayer, setConvLayer, setFlattenWay, setMiddleLayer, setParam, setInfoModal, setInfoName, trainInfo, setTrainInfo, augmentationParams, setAugmentationParams }) {
  const pList = ["kernel_size", "activation_function", "out_channel", "padding", "strid", "dropout_p", "neuron_size", "preprocessing", "way", "change_shape"]
  const { modelId, task } = useParams();
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

  const handleChangeParameter = async (key, value) => {
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
      
      // もし、keyがchange_shapeでpreprocessingがZCAの時
      if (key === 'change_shape' && newInputLayer['preprocessing'] === 'ZCA') {
        console.log('変更なし');
      } else {
        newInputLayer[key] = value;
        if (key === 'change_shape') {
          newInputLayer['shape'][0] = Number(value)
          newInputLayer['shape'][1] = Number(value)
        }
      }
      // もしkeyがpreprocessingでvalueがZCAの時、change_shapeとshapeをorigin_shapeに変更する
      if (key === 'preprocessing' && value === 'ZCA') {
        const originShape = await getOriginShape(modelId);
        newInputLayer['change_shape'] = Number(originShape);
        newInputLayer['shape'][0] = Number(originShape);
        newInputLayer['shape'][1] = Number(originShape);
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


  // データ拡張パラメータ
  const [sortTrainInfo, setSortTrainInfo] = useState(null);
  const [sortAugmentationParams, setSortAugmentationParams] = useState(null);
  const [information, setInformation] = useState(false);
  const [paramNames, setParamNames] = useState('');

  // 学習パラメータのソート
  useEffect(() => {
    const sortObject = () => {
      const sortedKeys = Object.keys(trainInfo).sort();
      const sortedObj = {};
      for (const key of sortedKeys) {
        sortedObj[key] = trainInfo[key];
      }
      setSortTrainInfo(sortedObj);
    };
    if (trainInfo) {
      sortObject();
    }
  }, [trainInfo]);


  // データ拡張パラメータのソート
  useEffect(() => {
    const sortObject = () => {
      const sortedKeys = Object.keys(augmentationParams).sort();
      const sortedObj = {};
      for (const key of sortedKeys) {
        sortedObj[key] = augmentationParams[key];
      }
      setSortAugmentationParams(sortedObj);
    };
    if (augmentationParams) {
      sortObject();
    }
  }, [augmentationParams]);

  const handleChangeAugmentationParameter = (key, value) => {
    const newParams = { ...augmentationParams };
    newParams[key] = value;
    setAugmentationParams(newParams);
  };

  return (
    <div className='edit-tile-parameter-wrapper'>
      <div className='edit-tile-field'>

        {param && keys.map((key, index) => (
          pList.includes(String(key)) && (
            <div key={index} >
              {key === 'change_shape' && inputLayer['preprocessing'] === 'ZCA' ? (
                null
              ) : (
                <div className='edit-wrapper'>
                  <div className='edit-field'>
                    <div className='param-information-button'>
                      <div className='button-wrapper' onClick={() => handleInfoModal(key)} style={{ cursor: 'pointer' }}>
                        <InfoIcon className='info-icon' />
                      </div>
                    </div>
                    <EditTileParamet name={key} value={param[key]} handleChangeParameter={handleChangeParameter} />
                  </div>
                </div>
              )}
            </div>
          )
        ))}

        {task === 'ImageClassification' && layerType === 'Input' && (
          <div className='train-panel-wrapper'>
            <TrainPanelTital title={'データ拡張パラメータ'} />
            <div className='panel-field'>
              {sortAugmentationParams && Object.entries(sortAugmentationParams).map(([key, value], index) => (
                <div key={index}>
                  <TrainPanelEdit
                    parameter={key}
                    value={value}
                    handleChangeParameter={handleChangeAugmentationParameter}
                    setInformation={setInformation}
                    setParamNames={setParamNames}
                  />
                </div>
              ))}
            </div>
            {information && <InformationModal infoName={paramNames} handleDelete={setInformation} />}
          </div>
        )}
      </div>
    </div>
  )
}

export default EditTileParameterField
