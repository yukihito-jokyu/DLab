import React, { useContext } from 'react'
import CNNDnDFild from '../utils/CNNDnDFild'
import FlappyConvStyle from '../FlappyBird/FlappyConvStyle'
import { CNNContext } from '../../page/ImageRecognition'

function MiddleLayer() {
  const { trainInfo } = useContext(CNNContext);
  const [convList, setConvList] = trainInfo.convLayer;
  return (
    <div>
      <h1>畳み込み層</h1>
      <CNNDnDFild middleLayer={FlappyConvStyle} middleData={[convList, setConvList]}  />
      <h1>全結合層</h1>
    </div>
  )
}

export default MiddleLayer
