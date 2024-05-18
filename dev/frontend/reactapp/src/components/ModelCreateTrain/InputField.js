import React from 'react';
import './InputField.css';
import InputTile from './InputTile';

function InputField() {
  return (
    <div className='input-field-wrapper'>
      <div className='input-tile-position'>
        <InputTile />
      </div>
    </div>
  )
};

export default InputField;
