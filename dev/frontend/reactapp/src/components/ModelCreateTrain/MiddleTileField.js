import React, { useState } from 'react'
import './ModelCreateTrain.css';
import NuronTile from './Tile/NuronTile';
import { ReactComponent as TileAdd } from '../../assets/svg/tile_add.svg'
import Conv2dTile from './Tile/Conv2dTile';
import Tile from './Tile/Tile';

function MiddleTileField({ tileName, layer ,setLayer, index, setNowIndex, handleModal }) {
  const handleAddTile = () => {
    setNowIndex(index+1);
    handleModal();
  };
  return (
    <div className='over-wrapper'>
      <div className='middle-tile-field-wrapper'>
        <div className='tile-position-wrapper'>
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
        <div className='delete-tile'></div>
      </div>
    </div>
  )
}

export default MiddleTileField
