import React from 'react';
import './OutputField.css';
import OutputTile from './OutputTile';

function OutputField() {
  return (
    <div className='output-field-wrapper'>
      <div className='output-tile-position'>
        <OutputTile />
      </div>
    </div>
  )
}

export default OutputField
