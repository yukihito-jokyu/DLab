import React, { useContext } from 'react'

import FlappyConvStyle from './FlappyConvStyle'
import CNNDnDFild from '../utils/CNNDnDFild'
import { CNNContext } from '../../page/Flappybird'

function FlappyConvLayer() {
  const { trainInfo } = useContext(CNNContext);
  const [convLayer, setConvList] = trainInfo.convLayer;
  return (
    <div>
      <h1>畳み込み層</h1>
      <div id='Conv-structure'>
        <CNNDnDFild middleLayer={FlappyConvStyle} middleData={[convLayer, setConvList]} />
      </div>
    </div>
  )
}

export default FlappyConvLayer
