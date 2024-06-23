import React from 'react'
import './Conv2dTile.css'

function Conv2dTile() {
  return (
    <div className='conv2d-tile-wrapper'>
      <div className='conv2d-tile'>
        <div className='tile-title-wrapper'>
          <p className='tile-title'>Conv2d</p>
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

export default Conv2dTile
