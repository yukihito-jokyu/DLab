import React from 'react'
import '../css/ImageProjectIcon.css'
import ImageProjectTitleFeild from './ImageProjectTitleFeild'
import ImageProjectImages from './ImageProjectImages'

function ImageProjectIcon() {
  return (
    <div className='ImageProjectIcon-border'>
      <div className='ImageProjectIcon-wrapper'>
        <ImageProjectTitleFeild />
        <ImageProjectImages />
      </div>
    </div>
  )
}

export default ImageProjectIcon
