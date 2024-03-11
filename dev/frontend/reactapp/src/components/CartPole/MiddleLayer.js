import React from 'react'
import DnDFild from '../utils/DnDFild';
import MiddleLayerStyle from './MiddleLayerStyle'

function MiddleLayer() {
  return (
    <div id='structure'>
      <h1>中間層</h1>
      <DnDFild middleLayer={MiddleLayerStyle} />
    </div>
  )
};

export default MiddleLayer;
