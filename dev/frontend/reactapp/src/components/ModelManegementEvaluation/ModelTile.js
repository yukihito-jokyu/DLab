import React from 'react';
import './ModelManegementEvaluation.css';
import { ReactComponent as PictureIcon } from '../../assets/svg/graph_24.svg'

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
          <div className='model-picture-icon-wrapper'>
            <PictureIcon className='picture-svg' />
          </div>
        </div>
      </div>
      <div className='graph-field'>
        <p>Graph</p>
        <div className='model-picture-filed-wrapper'>
          
        </div>
      </div>
    </div>
  )
}

export default ModelTile
