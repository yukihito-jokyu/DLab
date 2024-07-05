import React, { useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import EditTileParamet from './EditTileParamet';
import { v4 as uuidv4 } from 'uuid';

function EditTileParameterField({ parameter, convLayer, middleLayer, layerType, selectedindex, setConvLayer, setMiddleLayer }) {
  const pList =["kernel_size", "activ_func", "out_channel", "padding", "strid", "dropout_p", "input_size"]
  const [keys, setKeys] = useState([]);
  const [param, setParam] = useState(null);
  useEffect(() => {
    const handleSetParameter = () => {
      console.log('実行された')
      if (layerType === "Conv") {
        setParam(convLayer[selectedindex])
        const keys = Object.keys(convLayer[selectedindex]);
        const sortedKeys = keys.sort((a, b) => a.localeCompare(b));
        setKeys(sortedKeys);
      } else if (layerType === "Middle") {
        setParam(middleLayer[selectedindex])
        const keys = Object.keys(middleLayer[selectedindex]);
        const sortedKeys = keys.sort((a, b) => a.localeCompare(b));
        setKeys(sortedKeys);
      }
    }
    handleSetParameter();
  }, [parameter, layerType, selectedindex, convLayer, middleLayer]);
  const handleChangeParameter = (key, value) => {
    console.log(key, value)
    if (layerType === 'Conv') {
      const newConvLayer = [...convLayer];
      newConvLayer[selectedindex] = { ...newConvLayer[selectedindex], [key]: value}
      setConvLayer(newConvLayer);
    } else if (layerType === 'Middle') {
      const newMiddleLayer = [...middleLayer];
      newMiddleLayer[selectedindex] = { ...newMiddleLayer[selectedindex], [key]: value}
      setMiddleLayer(newMiddleLayer);
    }
  };
  return (
    <div className='edit-tile-parameter-wrapper'>
      <div className='edit-tile-field'>
        
        {param && keys.map((key, index) => (
          pList.includes(String(key)) && (
            <div key={index}>
              <EditTileParamet name={key} value={param[key]} handleChangeParameter={handleChangeParameter} />
            </div>
          )
        ))}
      </div>
    </div>
  )
}

export default EditTileParameterField
