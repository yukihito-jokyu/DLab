import React from 'react';
import './ModelCreateTrain.css';
import OutputTile from './Tile/OutputTile';

function OutputField({ outputShape }) {
  return (
    <div className='output-field-wrapper'>
      <div className='output-tile-position'>
        <OutputTile shape={outputShape} />
      </div>
    </div>
  )
}

export default OutputField
