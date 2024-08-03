import React, { useCallback, useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import EditScreen from './EditScreen';
import DataScreen from './DataScreen';
import EditTileParameterField from './EditTileParameterField';
import TrainLogField from './TrainLogField';
import { getModelStructure, getTrainInfo, updateStructure, updateTrainInfo } from '../../db/firebaseFunction';
import MiddleLayer from '../Image/MiddleLayer';
import TrainModal from './TrainModal';

function ScreenField({ edit, train, changeTrain }) {
  const [parameter, setParameter] = useState(null);
  const [param, setParam] = useState(null);
  const [parameterSet, setParameterSet] = useState(null);
  const [layerType, setLayerType] = useState(null);
  const [selectedindex, setSelectedIndex] = useState(null);
  const [inputLayer, setInputLayer] = useState('');
  const [convLayer, setConvLayer] = useState([]);
  const [flattenWay, setFlattenWay] = useState('');
  const [middleLayer, setMiddleLayer] = useState([]);
  const [outputLayer, setOutputLayer] = useState('');
  const [inputShape, setInputShape] = useState('');
  const [convShape, setConvShape] = useState([]);
  const [flattenShape, setFlattenShape] = useState([]);
  const [middleShape, setMiddleShape] = useState([]);
  const [outputShape, setOutputShape] = useState([]);
  const [trainInfo, setTrainInfo] = useState(null);
  const modelId = JSON.parse(sessionStorage.getItem('modelId'));
  const projectId = JSON.parse(sessionStorage.getItem('projectId'));
  
  // モデルの構造データ取得
  useEffect(() => {
    const fetchStructure = async () => {
      const structure = await getModelStructure(modelId);
      setInputLayer(structure.InputLayer);
      setMiddleLayer(structure.MiddleLayer);
      setFlattenWay(structure.FlattenWay);
      setConvLayer(structure.ConvLayer);
      setOutputLayer(structure.OutputLayer);
    };
    fetchStructure()
  }, [modelId]);

  // モデルの訓練情報取得
  useEffect(() => {
    const fetchTrainInfo = async () => {
      const info = await getTrainInfo(modelId);
      setTrainInfo(info);
    }
    fetchTrainInfo();
  }, [modelId]);

  // モデルの訓練情報の更新
  useEffect(() => {
    const saveTrainInfo = async () => {
      updateTrainInfo(modelId, trainInfo);
    };
    saveTrainInfo();
  }, [modelId, trainInfo]);


  useEffect(() => {
    // 形状の計算
    const shapeUpdate = () => {
      if (inputLayer.shape) {
        if (projectId === 'CartPole') {
          let N = inputLayer.shape;
          setInputShape([N])
          const newMiddleShape = []
          middleLayer.forEach((middle, index) => {
            if (middle.input_size) {
              N = middle.input_size;
            }
            newMiddleShape.push([N]);
          })
          setMiddleShape(newMiddleShape);
          N = outputLayer;
          setOutputShape([N]);
        } else {
          const inputShape = inputLayer.shape;
          let H = inputShape[0]
          let W = inputShape[1]
          let C = inputShape[2]
          let N = 0;
          let errorHandle = false;
          setInputShape([H, W, C]);
          const newConvShape = []
          convLayer.forEach((conv, index) => {
            const layerType = conv.layer_type;
            const kernelSize = conv.kernel_size;
            const padding = conv.padding;
            const strid = conv.strid;
            const outChannel = conv.out_channel;
            if (layerType === 'Conv2d' || layerType === 'MaxPool2d') {
              H = Math.trunc(((H - kernelSize + 2 * padding) / strid) + 1)
              W = Math.trunc(((W - kernelSize + 2 * padding) / strid) + 1)
              if (outChannel) {
                C = outChannel
              }
            }
            if (H < 1 || W < 1) {
              newConvShape.push(['error'])
              errorHandle = true
            } else {
              newConvShape.push([H, W, C]);
            }
            
          });
          setConvShape(newConvShape);
          const way = flattenWay.way;
          if (way === 'GAP' || way === 'GMP') {
            N = C;
          } else {
            N = H * W * C;
          }
          if (errorHandle) {
            setFlattenShape(['error']);
          } else {
            setFlattenShape([N])
          }
          
          const newMiddleShape = []
          middleLayer.forEach((middle, index) => {
            if (errorHandle) {
              newMiddleShape.push(['error'])
            } else {
              if (middle.input_size) {
                N = middle.input_size
              }
              newMiddleShape.push([N]);
            }
          })
          setMiddleShape(newMiddleShape);
          N = outputLayer;
          if (errorHandle) {
            setOutputShape(['error'])
          } else {
            setOutputShape([N]);
          }
        }
        
      }
    }
    shapeUpdate();
  }, [projectId, inputLayer, convLayer, flattenWay, middleLayer, outputLayer])

  // パラメータやタイルの位置の変更があったらfirebaseに保存する。
  useEffect(() => {
    const saveStructure = () => {
      // console.log(inputLayer)
      let structure = {}
      if (projectId === 'CartPole') {
        structure = {
          InputLayer: inputLayer,
          MiddleLayer: middleLayer,
          OutputLayer: outputLayer
        }
      } else {
        structure = {
          InputLayer: inputLayer,
          ConvLayer: convLayer,
          FlattenWay: flattenWay,
          MiddleLayer: middleLayer,
          OutputLayer: outputLayer
        }
      }
      
      updateStructure(modelId, structure);
    };
    
    saveStructure();
  }, [projectId, modelId, inputLayer, convLayer, flattenWay, middleLayer, outputLayer]);
  return (
    <div className='screen-field-wrapper'>
      <div className='left-screen'>
        {edit && (<EditScreen
          setParameter={setParameter}
          inputLayer={inputLayer}
          convLayer={convLayer}
          flattenWay={flattenWay}
          middleLayer={middleLayer}
          outputLayer={outputLayer}
          setConvLayer={setConvLayer}
          setMiddleLayer={setMiddleLayer}
          setParameterSet={setParameterSet}
          setLayerType={setLayerType}
          setSelectedIndex={setSelectedIndex}
          setParam={setParam}
          inputShape={inputShape}
          convShape={convShape}
          flattenShape={flattenShape}
          middleShape={middleShape}
          outputShape={outputShape}
        />)}
        {!edit && (<TrainLogField
          trainInfo={trainInfo}
          setTrainInfo={setTrainInfo}
        />)}
      </div>
      <div className='right-screen'>
        <div className='top-screen'>
          <DataScreen />
        </div>
        <div className='bottom-screen'>
          <EditTileParameterField
            parameter={parameter}
            inputLayer={inputLayer}
            convLayer={convLayer}
            flattenWay={flattenWay}
            middleLayer={middleLayer}
            layerType={layerType}
            param={param}
            selectedindex={selectedindex}
            setInputLayer={setInputLayer}
            setConvLayer={setConvLayer}
            setFlattenWay={setFlattenWay}
            setMiddleLayer={setMiddleLayer}
            setParam={setParam}
          />
        </div>
      </div>
      {train && (
        <TrainModal changeTrain={changeTrain} flattenShape={flattenShape} />
      )}
    </div>
  )
}

export default ScreenField
