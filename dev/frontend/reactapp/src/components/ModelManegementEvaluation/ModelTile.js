import React from 'react';
import './ModelTile.css';
import ModelPictureIcon from './ModelPictureIcon';
import ModelPictureField from './ModelPictureField';

function ModelTile() {
  return (
    <div className='model-tile-wrapper'>
      <div className='model-title-field'>
        <div className='model-check-box-wrapper'>
          <div className='model-check-box'></div>
        </div>
        <div className='model-title'>
          <p>Mnist_test1</p>
        </div>
        <div className='model-accuracy'>
          <p>0.76</p>
        </div>
        <div className='model-loss'>
          <p>0.044</p>
        </div>
        <div className='model-date'>
          <p>2024年3月26日</p>
        </div>
        <div className='model-picture'>
          <ModelPictureIcon />
        </div>
      </div>
      <div className='graph-field'>
        <p>Graph</p>
        <ModelPictureField />
      </div>
    </div>
  )
}

export default ModelTile
