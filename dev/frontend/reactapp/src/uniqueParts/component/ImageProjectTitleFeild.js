import React from 'react'
import '../css/ImageProjectTitleFeild.css'
import ImageProjectTitle from '../../uiParts/component/ImageProjectTitle'
import ImageProjectTitleRenameIcon from '../../uiParts/component/ImageProjectTitleRenameIcon'
import ImageProjectTitleLine from '../../uiParts/component/ImageProjectTitleLine'

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
