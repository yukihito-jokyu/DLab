import React from 'react';
import './ModelPictureIcon.css';
import { ReactComponent as PictureIcon } from '../../assets/svg/graph_24.svg'

function ModelPictureIcon() {
  return (
    <div className='model-picture-icon-wrapper'>
      <PictureIcon className='picture-svg' />
    </div>
  )
}

export default ModelPictureIcon
