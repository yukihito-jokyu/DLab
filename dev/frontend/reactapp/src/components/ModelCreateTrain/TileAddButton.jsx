import React from 'react';
import './TileAddButton.css';
import { ReactComponent as TileAdd } from '../../assets/svg/tile_add.svg'

function TileAddButton() {
  return (
    <div className='tile-add-button-wrapper'>
      <TileAdd className='tile-add-button' />
    </div>
  )
}

export default TileAddButton
