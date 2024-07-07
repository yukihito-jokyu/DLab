import React, { useCallback, useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import EditScreen from './EditScreen';
import DataScreen from './DataScreen';
import EditTileParameterField from './EditTileParameterField';
import TrainLogField from './TrainLogField';
import { getModelStructure, updateStructure } from '../../db/firebaseFunction';
import MiddleLayer from '../Image/MiddleLayer';

function ScreenField() {
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
  const modelId = JSON.parse(sessionStorage.getItem('modelId'));
  
  useEffect(() => {
    const fetchStructure = async () => {
      // const modelId = "model_test"
      
      const structure = await getModelStructure(modelId);
      setInputLayer(structure.InputLayer);
      setMiddleLayer(structure.MiddleLayer);
      setFlattenWay(structure.FlattenWay);
      setConvLayer(structure.ConvLayer);
      setOutputLayer(structure.OutputLayer);
    };

    fetchStructure()
  }, [modelId]);

  useEffect(() => {
    // 形状の計算
    const shapeUpdate = () => {
      if (inputLayer.shape) {
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
          const nuronNum = middle.input_size;
          N = nuronNum;
          if (errorHandle) {
            newMiddleShape.push(['error'])
          } else {
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
    shapeUpdate();
  }, [inputLayer, convLayer, flattenWay, middleLayer, outputLayer])

  // パラメータやタイルの位置の変更があったらfirebaseに保存する。
  useEffect(() => {
    
    
    const saveStructure = () => {
      const structure = {
        InputLayer: inputLayer,
        ConvLayer: convLayer,
        FlattenWay: flattenWay,
        MiddleLayer: middleLayer,
        OutputLayer: outputLayer
      }
      updateStructure(modelId, structure);
    };
    
    saveStructure();
  }, [modelId, inputLayer, convLayer, flattenWay, middleLayer, outputLayer]);
  return (
    <div className='screen-field-wrapper'>
      <div className='left-screen'>
        {/* <EditScreen
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
        /> */}
        <TrainLogField />
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
    </div>
  )
}

export default ScreenField
