import React, { useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import InputField from './InputField';
import OutputField from './OutputField';
import MiddleFeild from './MiddleFeild';
import ConvField from './ConvField';
import FlattenField from './FlattenField';
import TileAddModal from './TileAddModal';
import { v4 as uuidv4 } from 'uuid';
import ErrorModal from './ErrorModal';

function EditScreen({ setParameter, inputLayer, convLayer, flattenWay, middleLayer, outputLayer, setConvLayer, setMiddleLayer, setParameterSet, setLayerType, setSelectedIndex, setParam, inputShape, convShape, flattenShape, middleShape, outputShape }) {
  const [convAdd, setConvAdd] = useState(false);
  const [middleAdd, setMiddleAdd] = useState(false);
  const [nowIndex, setNowIndex] = useState(null);
  const [errorModal, setErrorModal] = useState(false);
  const [errorField, setErrorField] = useState('');
  const [errorTile, setErrorTile] = useState('');
  const handleConvModal = () => {
    setConvAdd(!convAdd);
  };
  const handleMiddleModal = () => {
    setMiddleAdd(!middleAdd);
  };
  const handleAddConvTile = (layerType) => {
    if (layerType === 'Conv2d') {
      const newLayer = {
        id: uuidv4(),
        activ_func: "ReLU",
        kernel_size: 3,
        layer_type: layerType,
        out_channel: 64,
        padding: 0,
        strid: 1,
        type: "Conv"
      };
      const copyLayer = [...convLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setConvLayer(copyLayer);
    } else if (layerType === 'MaxPool2d') {
      const newLayer = {
        id: uuidv4(),
        kernel_size: 3,
        layer_type: layerType,
        padding: 0,
        strid: 1,
        type: "Conv"
      };
      const copyLayer = [...convLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setConvLayer(copyLayer);
    } else if (layerType === 'Dropout') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType,
        dropout_p: 0.1,
        type: "Conv"
      };
      const copyLayer = [...convLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setConvLayer(copyLayer);
    } else if (layerType === 'BatchNorm') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType,
        type: "Conv"
      };
      const copyLayer = [...convLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setConvLayer(copyLayer);
    } else if (layerType === 'Neuron') {
      // const newLayer = {
      //   id: uuidv4(),
      //   layer_type: layerType,
      //   activ_func: "ReLU",
      //   input_size: 100,
      //   type: "Conv"
      // };
      // const copyLayer = [...convLayer];
      // copyLayer.splice(nowIndex, 0, newLayer);
      // setConvLayer(copyLayer);
      setErrorModal(!errorModal);
      setErrorField('畳み込み層');
      setErrorTile('Nuron');
    };
    setConvAdd(!convAdd);
  };
  const handleAddMiddleTile = (layerType) => {
    if (layerType === 'Conv2d') {
      // const newLayer = {
      //   id: uuidv4(),
      //   activ_func: "ReLU",
      //   kernel_size: 3,
      //   layer_type: layerType,
      //   out_channel: 64,
      //   padding: 0,
      //   strid: 1,
      //   type: "Middle"
      // };
      // const copyLayer = [...middleLayer];
      // copyLayer.splice(nowIndex, 0, newLayer);
      // setMiddleLayer(copyLayer);
      setErrorModal(!errorModal);
      setErrorField('全結合層');
      setErrorTile(layerType);
    } else if (layerType === 'MaxPool2d') {
      // const newLayer = {
      //   id: uuidv4(),
      //   kernel_size: 3,
      //   layer_type: layerType,
      //   padding: 0,
      //   strid: 1,
      //   type: "Middle"
      // };
      // const copyLayer = [...middleLayer];
      // copyLayer.splice(nowIndex, 0, newLayer);
      // setMiddleLayer(copyLayer);
      setErrorModal(!errorModal);
      setErrorField('全結合層');
      setErrorTile(layerType);
    } else if (layerType === 'Dropout') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType,
        dropout_p: 0.1,
        type: "Middle"
      };
      const copyLayer = [...middleLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setMiddleLayer(copyLayer);
    } else if (layerType === 'BatchNorm') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType,
        type: "Middle"
      };
      const copyLayer = [...middleLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setMiddleLayer(copyLayer);
    } else if (layerType === 'Neuron') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType,
        activ_func: "ReLU",
        input_size: 100,
        type: "Middle"
      };
      const copyLayer = [...middleLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setMiddleLayer(copyLayer);
    };
    setMiddleAdd(!middleAdd);
  };

  const handleDeleteConvTile = (index) => {
    const newLayer = [...convLayer];
    newLayer.splice(index, 1);
    setConvLayer(newLayer);
    setParam(null);
  };
  const handleDeleteMiddleTile = (index) => {
    const newLayer = [...middleLayer];
    newLayer.splice(index, 1);
    setMiddleLayer(newLayer);
    setParam(null)
  };

  const handleErrorModal = () => {
    setErrorModal(!errorModal);
  }
  return (
    <div className='edit-screen-wrapper'>
      <InputField
        inputLayer={inputLayer}
        setLayerType={setLayerType}
        shape={inputShape}
      />
      <ConvField
        convLayer={convLayer}
        setConvLayer={setConvLayer}
        setNowIndex={setNowIndex}
        handleModal={handleConvModal}
        handleDeleteConvTile={handleDeleteConvTile}
        setParameter={setParameter}
        setParameterSet={setParameterSet}
        setLayerType={setLayerType}
        setSelectedIndex={setSelectedIndex}
        convShape={convShape}
      />
      <FlattenField
        flattenWay={flattenWay}
        setLayerType={setLayerType}
        flattenShape={flattenShape}
      />
      <MiddleFeild
        middleLayer={middleLayer}
        setMiddleLayer={setMiddleLayer}
        setNowIndex={setNowIndex}
        handleModal={handleMiddleModal}
        handleDeleteMiddleTile={handleDeleteMiddleTile}
        setParameter={setParameter}
        setParameterSet={setParameterSet}
        setLayerType={setLayerType}
        setSelectedIndex={setSelectedIndex}
        middleShape={middleShape}
      />
      <OutputField
        outputShape={outputShape}
      />
      {convAdd && 
        <TileAddModal
          handleModal={handleConvModal}
          handleAddTile={handleAddConvTile}
        />
      }
      {middleAdd &&
        <TileAddModal
          handleModal={handleMiddleModal}
          handleAddTile={handleAddMiddleTile}
        />
      }
      {errorModal && <ErrorModal
        handleErrorModal={handleErrorModal}
        filedName={errorField}
        tileName={errorTile}
      />}
    </div>
  )
}

export default EditScreen
