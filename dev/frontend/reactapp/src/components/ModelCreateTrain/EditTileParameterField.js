import React, { useEffect, useState } from 'react';
import './ModelCreateTrain.css';
import EditTileParamet from './EditTileParamet';

function EditTileParameterField({ parameter }) {
  const [keys, setKeys] = useState([]);
  useEffect(() => {
    const handleP = () => {
      if (parameter !== null) {
        const keys = Object.keys(parameter);
        setKeys(keys);
      };
    };
    handleP();
  }, [parameter]);
  return (
    <div className='edit-tile-parameter-wrapper'>
      <div className='edit-tile-field'>
        
        {parameter && keys.map((key, index) => (
          <div key={index}>
            <EditTileParamet name={key} value={parameter[key]} />
          </div>
        ))}
      </div>
    </div>
  )
}

export default EditTileParameterField
