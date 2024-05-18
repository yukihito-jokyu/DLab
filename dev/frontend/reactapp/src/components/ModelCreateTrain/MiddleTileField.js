import React from 'react'
import './MiddleTileField.css';
import NuronTile from './NuronTile';
import TileAddButton from './TileAddButton';

function MiddleTileField() {
  return (
    <div className='middle-tile-field-wrapper'>
      <div className='delete-tile'></div>
      <div className='tile-position-wrapper'>
        <NuronTile />
        <TileAddButton />
      </div>
    </div>
  )
}

export default MiddleTileField
