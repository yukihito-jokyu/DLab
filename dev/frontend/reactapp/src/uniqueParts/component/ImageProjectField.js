import React from 'react'
import '../css/ImageProjectField.css'
import ImageProjectIcon from './ImageProjectIcon'
import ImageProjectAdd from './ImageProjectAdd'


function ImageProjectField() {
  return (
    <div className='imageprojectfield-wrapper'>
      <ImageProjectIcon />
      <ImageProjectAdd />
    </div>
  )
}

export default ImageProjectField
