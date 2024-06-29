import React, { useState } from 'react'
import './ModelCreateTrain.css';
import NuronTile from './Tile/NuronTile';
import { ReactComponent as TileAdd } from '../../assets/svg/tile_add.svg'
import Conv2dTile from './Tile/Conv2dTile';
import Tile from './Tile/Tile';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';

function MiddleTileField({ tileName, layer ,setLayer, index, setNowIndex, handleModal, handleDeleteTile, setParameter }) {
  const handleAddTile = () => {
    setNowIndex(index+1);
    handleModal();
  };
  return (
    <div className='over-wrapper'>
      <div className='middle-tile-field-wrapper'>
        <div className='delete-tile'>
          <div className='delete-button-wrapper'>
            <div className='delete-button' onClick={() => handleDeleteTile(index)}>
              <DeletIcon className='delet-svg' />
            </div>
          </div>
        </div>
        <div className='tile-position-wrapper' onClick={() => setParameter(layer)}>
          {tileName === 'Conv2d' ? (
            <Conv2dTile />
          ) : tileName === 'Neuron' ? (
            <NuronTile />
          ) : (
            <Tile text={tileName} />
          )}
          <div className='tile-add-button-wrapper'>
            <div onClick={(handleAddTile)}>
              <TileAdd className='tile-add-button' />
            </div>
          </div>
        </div>
        
      </div>
    </div>
  )
}

export default MiddleTileField
