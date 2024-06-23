import React from 'react';
import './NuronTile.css';

function NuronTile() {
  return (
    <div className='nuron-tile-wrapper'>
      <div className='nuron-tile'>
        <div className='tile-title-wrapper'>
          <p className='tile-title'>Nuron</p>
        </div>
        <div className='output-dim-wrapper'>
          <p>xxx,xxx,xxx</p>
        </div>
      </div>
      <div className='activate-tile-wrapper'>
        <div className='activate-tile'>
          <p className='activate'>ReLU</p>
        </div>
      </div>
    </div>
  )
}

export default NuronTile
