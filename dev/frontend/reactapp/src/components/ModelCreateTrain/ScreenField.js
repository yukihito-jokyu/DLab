import React, { useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import EditScreen from './EditScreen';
import DataScreen from './DataScreen';
import EditTileParameterField from './EditTileParameterField';
import TrainLogField from './TrainLogField';
import { getModelStructure } from '../../db/firebaseFunction';

function ScreenField() {
  const [parameter, setParameter] = useState(null);
  const [parameterSet, setParameterSet] = useState(null);
  const [layerType, setLayerType] = useState(null);
  const [selectedindex, setSelectedIndex] = useState(null);
  const [inputLayer, setInputLayer] = useState([]);
  const [convLayer, setConvLayer] = useState([]);
  const [flattenWay, setFlattenWay] = useState('');
  const [middleLayer, setMiddleLayer] = useState([]);
  const [outputLayer, setOutputLayer] = useState('');
  
  useEffect(() => {
    const fetchStructure = async () => {
      const modelId = "model_test"
      const structure = await getModelStructure(modelId);
      setInputLayer(structure.InputLayer);
      setMiddleLayer(structure.MiddleLayer);
      setFlattenWay(structure.FlattenWay);
      setConvLayer(structure.ConvLayer);
      setOutputLayer(structure.OutputLayer);
    };

    fetchStructure()
  }, []);
  return (
    <div className='screen-field-wrapper'>
      <div className='left-screen'>
        <EditScreen
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
        />
        {/* <TrainLogField /> */}
      </div>
      <div className='right-screen'>
        <div className='top-screen'>
          <DataScreen />
        </div>
        <div className='bottom-screen'>
          <EditTileParameterField
            parameter={parameter}
            convLayer={convLayer}
            middleLayer={middleLayer}
            layerType={layerType}
            selectedindex={selectedindex}
            setConvLayer={setConvLayer}
            setMiddleLayer={setMiddleLayer}
          />
        </div>
      </div>
    </div>
  )
}

export default ScreenField
