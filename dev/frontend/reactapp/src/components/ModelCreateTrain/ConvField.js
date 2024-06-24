import React from 'react'
import './ModelCreateTrain.css';
import MiddleTileField from './MiddleTileField';
import { ReactComponent as TileAdd } from '../../assets/svg/tile_add.svg'

function ConvField({ convLayer, setConvLayer, setNowIndex, handleModal, handleDeleteConvTile }) {
  const handleAddTile = () => {
    setNowIndex(0);
    handleModal();
  };
  return (
    <div className='middle-field-wrapper'>
      <div className='tile-add-button-over-wrapper'>
        <div className='tile-add-button-wrapper'>
          <div onClick={(handleAddTile)}>
            <TileAdd className='tile-add-button' />
          </div>
        </div>
      </div>
      {convLayer.map((conv, index) => (
        <MiddleTileField key={index} tileName={conv.layer_type} layer={convLayer} setLayer={setConvLayer} index={index} setNowIndex={setNowIndex} handleModal={handleModal} handleDeleteConvTile={handleDeleteConvTile} />
        // console.log(index)
      ))}
    </div>
  )
}

export default ConvField;
