import React from 'react'
import './ImageProjectIcon.css'
import ImageProjectTitleFeild from './ImageProjectTitleFeild'
import ImageProjectImages from './ImageProjectImages'

function ImageProjectIcon() {
  return (
    <div className='ImageProjectIcon-wrapper'>
      <ImageProjectTitleFeild />
      <ImageProjectImages />
    </div>
  )
}

export default ImageProjectIcon