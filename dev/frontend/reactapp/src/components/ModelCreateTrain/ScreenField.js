import React, { useCallback, useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import EditScreen from './EditScreen';
import DataScreen from './DataScreen';
import EditTileParameterField from './EditTileParameterField';
import TrainLogField from './TrainLogField';
import MiddleLayer from '../Image/MiddleLayer';
import TrainModal from './TrainModal';
import { getModelStructure, getTrainInfo, updateStructure, updateTrainInfo } from '../../db/function/model_structure';
import { useParams } from 'react-router-dom';
import InformationModal from './Modal/InformationModal';
import TrainPanel from './TrainPanel';
import VisImageModal from './VisImageModal';
import VisTrainModal from './VisTrainModal';
import { socket } from '../../socket/socket';

function ScreenField({ edit, train, visImageModal, visTrainModal, changeTrain, changeVisImageModal, changeVisTrainModal }) {
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
  const [infoModal, setInfoModal] = useState(false);
  const [infoName, setInfoName] = useState();
  const { projectName, modelId } = useParams();
  const [images, setImages] = useState([]);   // 学習結果の保管場所 
  const [originImages, setOriginImages] = useState([]);   // 画像分類の学習結果保管場所
  const [labels, setLabels] = useState([]);
  const [preLabels, setPreLabels] = useState([]);
  const [j, setJ] = useState(0);
  const [i, setI] = useState(0);

  // モデルの構造データ取得
  useEffect(() => {
    const fetchStructure = async () => {
      const structure = await getModelStructure(modelId);
      setInputLayer(structure.input_layer);
      setMiddleLayer(structure.middle_layer);
      setFlattenWay(structure.flatten_method);
      setConvLayer(structure.conv_layer);
      setOutputLayer(structure.output_layer);
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
        if (projectName === 'CartPole') {
          let N = inputLayer.shape;
          setInputShape([N])
          const newMiddleShape = []
          middleLayer.forEach((middle, index) => {
            if (middle.neuron_size) {
              N = middle.neuron_size;
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
              if (middle.neuron_size) {
                N = middle.neuron_size
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
  }, [projectName, inputLayer, convLayer, flattenWay, middleLayer, outputLayer])

  // パラメータやタイルの位置の変更があったらfirebaseに保存する。
  useEffect(() => {
    const saveStructure = () => {
      // console.log(inputLayer)
      let structure = {}
      if (projectName === 'CartPole') {
        structure = {
          input_layer: inputLayer,
          middle_layer: middleLayer,
          output_layer: outputLayer
        }
      } else {
        structure = {
          input_layer: inputLayer,
          conv_layer: convLayer,
          flatten_method: flattenWay,
          middle_layer: middleLayer,
          output_layer: outputLayer
        }
      }

      updateStructure(modelId, structure);
    };

    saveStructure();
  }, [projectName, modelId, inputLayer, convLayer, flattenWay, middleLayer, outputLayer]);


  // 学習結果の取得
  useEffect(() => {
    const handleSetImage = (response) => {
      setImages(response.Images)
      setI(0)
      console.log(response.Images)
    }

    const handleOriginImage = (response) => {
      console.log(response)
      setOriginImages(response.Images);
      setLabels(response.Labels);
      setPreLabels(response.PreLabels);
      setJ(0)
    }

    const handleCleaner = (respose) => {
      setImages([]);
      setOriginImages([]);
      setLabels([]);
      setPreLabels([]);
      setI(0);
      setJ(0);
    }

    // イベントの発火
    socket.on('flappy_valid' + modelId, handleSetImage);
    socket.on('cartpole_valid' + modelId, handleSetImage);
    socket.on('image_valid' + modelId, handleOriginImage);

    socket.on('flappy_train_end' + modelId, handleCleaner);
    socket.on('cartpole_train_end' + modelId, handleCleaner);
    socket.on('image_train_end' + modelId, handleCleaner);

    // クリーンアップ関数を返す
    return () => {
      socket.off('flappy_valid' + modelId, handleSetImage);
      socket.off('cartpole_valid' + modelId, handleSetImage);
      socket.off('image_valid' + modelId, handleOriginImage);

      socket.off('flappy_train_end' + modelId, handleCleaner);
      socket.off('cartpole_train_end' + modelId, handleCleaner);
      socket.off('image_train_end' + modelId, handleCleaner);
    };

  }, [modelId, changeVisImageModal, changeVisTrainModal]);

  // 0.1秒ごとにiをインクリメントする
  useEffect(() => {
    const interval = setInterval(() => {
      setI((PrevI) => {
        if (PrevI >= images.length - 1) {
          return 0
        }
        return PrevI + 1;
      });
    }, 100) // 100ms = 0.1秒
    
    // クリーンアップ関数でインターバルをクリア
    return () => clearInterval(interval)
  }, [images]);

  // 0.1秒ごとにiをインクリメントする
  useEffect(() => {
    const interval = setInterval(() => {
      setJ((PrevI) => {
        if (PrevI >= originImages.length - 1) {
          return 0
        }
        return PrevI + 1;
      });
    }, 3000) // 3000ms = 3秒
    
    // クリーンアップ関数でインターバルをクリア
    return () => clearInterval(interval)
  }, [originImages]);

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
        {!edit && (<TrainLogField />)}
      </div>
      {edit && (<div className='right-screen'>
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
            setInfoModal={setInfoModal}
            setInfoName={setInfoName}
          />
        </div>
      </div>)}
      {!edit && (
        <div className='right-screen-param'>
          <TrainPanel
            trainInfo={trainInfo}
            setTrainInfo={setTrainInfo}
          />
        </div>
      )}
      {train && (
        <TrainModal changeTrain={changeTrain} flattenShape={flattenShape} />
      )}
      {infoModal && (
        <InformationModal infoName={infoName} handleDelete={setInfoModal} />
      )}
      {visImageModal && (
        <VisImageModal changeVisImageModal={changeVisImageModal} image={originImages[j]} label={labels[j]} preLabel={preLabels[j]} />
      )}
      {visTrainModal && (
        <VisTrainModal changeVisTrainModal={changeVisTrainModal} image={images[i]} />
      )}
    </div>
  )
}

export default ScreenField;
