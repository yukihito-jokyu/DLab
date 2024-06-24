import React, { useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import InputField from './InputField';
import OutputField from './OutputField';
import MiddleFeild from './MiddleFeild';
import ConvField from './ConvField';
import { getModelStructure } from '../../db/firebaseFunction';
import FlattenField from './FlattenField';
import TileAddModal from './TileAddModal';

function EditScreen() {
  const [inputLayer, setInputLayer] = useState([]);
  const [convLayer, setConvLayer] = useState([]);
  const [flattenWay, setFlattenWay] = useState('');
  const [middleLayer, setMiddleLayer] = useState([]);
  const [outputLayer, setOutputLayer] = useState('');
  const [add, setAdd] = useState(false);
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
  const handleModal = () => {
    setAdd(!add);
  };
  const handleAddConvTile = (layerType) => {
    if (layerType === 'Conv2d') {
      const newLayer = {
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
        layer_type: layerType,
        dropout_p: 0.1,
      };
      const copyLayer = [...convLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setConvLayer(copyLayer);
    } else if (layerType === 'BatchNorm') {
      const newLayer = {
        layer_type: layerType
      };
      const copyLayer = [...convLayer];
      copyLayer.splice(nowIndex, 0, newLayer);
      setConvLayer(copyLayer);
    }
    setAdd(!add);
  }

  const handleDeleteConvTile = (index) => {
    const newLayer = [...convLayer];
    newLayer.splice(index, 1);
    setConvLayer(newLayer);
  }
  return (
    <div className='edit-screen-wrapper'>
      <InputField  />
      <ConvField convLayer={convLayer} setConvLayer={setConvLayer} setNowIndex={setNowIndex} handleModal={handleModal} handleDeleteConvTile={handleDeleteConvTile} />
      <FlattenField />
      <MiddleFeild middleLayer={middleLayer} />
      <OutputField />
      {add && <TileAddModal handleModal={handleModal} handleAddTile={handleAddConvTile} />}
    </div>
  )
}

export default EditScreen
