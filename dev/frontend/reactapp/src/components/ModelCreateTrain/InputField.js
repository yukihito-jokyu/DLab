import React from 'react';
import './ModelCreateTrain.css';
import InputTile from './Tile/InputTile';
import Tile from './Tile/Tile';

function InputField() {
  const text = 'Input'
  return (
    <div className='input-field-wrapper'>
      <div className='input-tile-position'>
        <Tile text={text} />
      </div>
    </div>
  )
};

export default InputField;
