import React from 'react';
import './ModelCreateTrain.css';

function DataScreen() {
  return (
    <div className='data-screen-wrapper'>
      <div className='image-title-wrapper'>
        <p>Image A</p>
      </div>
      <div className='image-data-wrapper'>
        <div className='image-before-data'></div>
        <div className='image-arrow'>
          <p>▶</p>
        </div>
        <div className='image-after-data'></div>
      </div>
      <div className='image-status-wrapper'>
        <div className='image-before'>
          <p>Before</p>
        </div>
        <div className='image-after'>
          <p>After</p>
        </div>
      </div>
    </div>
  )
}

export default DataScreen
