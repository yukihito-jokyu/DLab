import React from 'react'
import './ImageProjectImages.css'
import ImageFeild from './ImageFeild'

function ImageProjectImages() {
  return (
    <div className='projectimages-wrapper'>
      <div className='big-image'>
        <ImageFeild elementsize={150} />
      </div>
      <div className='small-image'>
        <div className='is-first'>
          <ImageFeild elementsize={70} />
        </div>
        <div>
          <ImageFeild elementsize={70} />
        </div>
      </div>
    </div>
  )
}

export default ImageProjectImages
