import React from 'react';
import './ModelCreateTrain.css';
import Tile from './Tile/Tile';

function InputField({ inputLayer, setLayerType, shape }) {
  const text = inputLayer.type;
  const handleClick = () => {
    setLayerType(inputLayer.type);
  }
  return (
    <div className='input-field-wrapper'>
      <div className='input-tile-position'>
        <div onClick={handleClick} style={{ cursor: 'pointer' }}>
          <Tile text={text} shape={shape} />
        </div>
      </div>
    </div>
  )
};

export default InputField;
