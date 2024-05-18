import React from 'react';
import './ModelDeletIcon.css';
import { ReactComponent as DeleteIcon } from '../../assets/svg/delete_24.svg'

function ModelDeletIcon() {
  return (
    <div className='model-delet-icon-wrapper'>
      <DeleteIcon className='model-delet-svg' />
    </div>
  )
}

export default ModelDeletIcon
