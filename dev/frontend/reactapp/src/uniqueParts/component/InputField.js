import React from 'react';
import '../css/InputField.css';
import InputTile from '../../uiParts/component/InputTile';

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
