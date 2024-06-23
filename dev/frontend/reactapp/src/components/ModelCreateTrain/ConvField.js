import React from 'react'
import './ModelCreateTrain.css';
import MiddleTileField from './MiddleTileField';

function ConvField({ convLayer, setConvLayer, setNowIndex, handleModal }) {
  return (
    <div className='middle-field-wrapper'>
      {convLayer.map((conv, index) => (
        <MiddleTileField key={index} tileName={conv.layer_type} layer={convLayer} setLayer={setConvLayer} index={index} setNowIndex={setNowIndex} handleModal={handleModal} />
        // console.log(index)
      ))}
    </div>
  )
}

export default ConvField;
