import React, { useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import InputField from './InputField';
import OutputField from './OutputField';
import MiddleFeild from './MiddleFeild';
import ConvField from './ConvField';
import { getModelStructure } from '../../db/firebaseFunction';
import FlattenField from './FlattenField';
import TileAddModal from './TileAddModal';
import { v4 as uuidv4 } from 'uuid';

function EditScreen({ setParameter }) {
  const [inputLayer, setInputLayer] = useState([]);
  const [convLayer, setConvLayer] = useState([]);
  const [flattenWay, setFlattenWay] = useState('');
  const [middleLayer, setMiddleLayer] = useState([]);
  const [outputLayer, setOutputLayer] = useState('');
  const [convAdd, setConvAdd] = useState(false);
  const [middleAdd, setMiddleAdd] = useState(false);
  const [nowIndex, setNowIndex] = useState(null);
  useEffect(() => {
    const fetchStructure = async () => {
      const modelId = "model_test"
      const structure = await getModelStructure(modelId);
      setInputLayer(structure.InputLayer);
      setMiddleLayer(structure.MiddleLayer);
      setFlattenWay(structure.FlattenWay);
      setConvLayer(structure.ConvLayer);
      setOutputLayer(structure.OutputLayer);
      console.log(structure.InputLayer);
    };

    fetchStructure()
  }, []);
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
        strid: 1
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
        strid: 1
      };
      const copyLayer = [...convLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setConvLayer(copyLayer);
    } else if (layerType === 'Dropout') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType,
        dropout_p: 0.1,
      };
      const copyLayer = [...convLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setConvLayer(copyLayer);
    } else if (layerType === 'BatchNorm') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType
      };
      const copyLayer = [...convLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setConvLayer(copyLayer);
    } else if (layerType === 'Neuron') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType,
        activ_func: "ReLU",
        input_size: 100
      };
      const copyLayer = [...convLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setConvLayer(copyLayer);
    };
    setConvAdd(!convAdd);
  };
  const handleAddMiddleTile = (layerType) => {
    if (layerType === 'Conv2d') {
      const newLayer = {
        id: uuidv4(),
        activ_func: "ReLU",
        kernel_size: 3,
        layer_type: layerType,
        out_channel: 64,
        padding: 0,
        strid: 1
      };
      const copyLayer = [...middleLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setMiddleLayer(copyLayer);
    } else if (layerType === 'MaxPool2d') {
      const newLayer = {
        id: uuidv4(),
        kernel_size: 3,
        layer_type: layerType,
        padding: 0,
        strid: 1
      };
      const copyLayer = [...middleLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setMiddleLayer(copyLayer);
    } else if (layerType === 'Dropout') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType,
        dropout_p: 0.1,
      };
      const copyLayer = [...middleLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setMiddleLayer(copyLayer);
    } else if (layerType === 'BatchNorm') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType
      };
      const copyLayer = [...middleLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setMiddleLayer(copyLayer);
    } else if (layerType === 'Neuron') {
      const newLayer = {
        id: uuidv4(),
        layer_type: layerType,
        activ_func: "ReLU",
        input_size: 100
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
  };
  const handleDeleteMiddleTile = (index) => {
    const newLayer = [...middleLayer];
    newLayer.splice(index, 1);
    setMiddleLayer(newLayer);
  };
  return (
    <div className='edit-screen-wrapper'>
      <InputField  />
      <ConvField convLayer={convLayer} setConvLayer={setConvLayer} setNowIndex={setNowIndex} handleModal={handleConvModal} handleDeleteConvTile={handleDeleteConvTile} setParameter={setParameter} />
      <FlattenField />
      <MiddleFeild middleLayer={middleLayer} setMiddleLayer={setMiddleLayer} setNowIndex={setNowIndex} handleModal={handleMiddleModal} handleDeleteMiddleTile={handleDeleteMiddleTile} setParameter={setParameter} />
      <OutputField />
      {convAdd && <TileAddModal handleModal={handleConvModal} handleAddTile={handleAddConvTile} />}
      {middleAdd && <TileAddModal handleModal={handleMiddleModal} handleAddTile={handleAddMiddleTile} />}
    </div>
  )
}

export default EditScreen
