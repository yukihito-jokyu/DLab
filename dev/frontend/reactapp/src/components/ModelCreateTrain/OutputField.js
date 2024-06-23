import React from 'react';
import './ModelCreateTrain.css';
import OutputTile from './Tile/OutputTile';

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
