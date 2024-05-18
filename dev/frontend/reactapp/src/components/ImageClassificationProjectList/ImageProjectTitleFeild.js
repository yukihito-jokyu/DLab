import React from 'react'
import './ImageProjectTitleFeild.css'
import ImageProjectTitle from './ImageProjectTitle'
import ImageProjectTitleRenameIcon from './ImageProjectTitleRenameIcon'
import ImageProjectTitleLine from './ImageProjectTitleLine'

function ImageProjectTitleFeild() {
  return (
    <div className='titlefeild-wrapper'>
      <div className='project-title'>
        <ImageProjectTitle />
        <ImageProjectTitleRenameIcon />
      </div>
      <ImageProjectTitleLine />
    </div>
  )
}

export default ImageProjectTitleFeild
