import React from 'react'
import '../css/MiddleTileField.css';
import NuronTile from '../../uiParts/component/NuronTile';
import TileAddButton from '../../uiParts/component/TileAddButton';

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
