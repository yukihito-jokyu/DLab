import React, { useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import EditTileParamet from './EditTileParamet';
import { v4 as uuidv4 } from 'uuid';

function EditTileParameterField({ parameter }) {
  const pList =["kernel_size", "activ_func", "out_channel", "padding", "strid", "dropout_p", "input_size"]
  const [keys, setKeys] = useState([]);
  useEffect(() => {
    const handleP = () => {
      if (parameter !== null) {
        const keys = Object.keys(parameter);
        const sortedKeys = keys.sort((a, b) => a.localeCompare(b));
        setKeys(sortedKeys);
      };
    };
    handleP();
  }, [parameter]);
  return (
    <div className='edit-tile-parameter-wrapper'>
      <div className='edit-tile-field'>
        
        {parameter && keys.map((key, index) => (
          pList.includes(String(key)) && (
            <div key={index}>
              <EditTileParamet name={key} value={parameter[key]} />
            </div>
          )
        ))}
      </div>
    </div>
  )
}

export default EditTileParameterField
