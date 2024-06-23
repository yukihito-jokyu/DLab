import React from 'react';
import './ModelCreateTrain.css';
import MiddleTileField from './MiddleTileField';

function MiddleFeild({ middleLayer }) {
  return (
    <div className='middle-field-wrapper'>
      {middleLayer.map((middle, index) => (
        <MiddleTileField key={index} tileName={middle.layer_type} />
      ))}
    </div>
  )
}

export default MiddleFeild
