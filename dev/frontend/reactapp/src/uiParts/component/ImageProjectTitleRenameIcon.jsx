import React from 'react'
import { ReactComponent as EditSVG } from '../../assets/svg/edit.svg' 
import '../css/ImageProjectTitleRenameIcon.css'

function ImageProjectTitleRenameIcon() {
  return (
    <div className='rename-icon'>
      <EditSVG className='edit-svg' />
    </div>
  )
}

export default ImageProjectTitleRenameIcon
