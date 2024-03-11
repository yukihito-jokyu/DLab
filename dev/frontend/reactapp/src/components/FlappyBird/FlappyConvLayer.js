import React from 'react'

import FlappyConvStyle from './FlappyConvStyle'
import CNNDnDFild from '../utils/CNNDnDFild'

function FlappyConvLayer() {
  return (
    <div>
      <h1>畳み込み層</h1>
      <div id='Conv-structure'>
        <CNNDnDFild middleLayer={FlappyConvStyle} />
      </div>
    </div>
  )
}

export default FlappyConvLayer
