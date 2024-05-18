import React from 'react';
import './ModelCreateButton.css';
import { ReactComponent as AddIcon } from '../../assets/svg/project_add_48.svg'

function ModelCreateButton() {
  return (
    <div className='model-create-button-wrapper'>
      <AddIcon className='model-add-svg' />
    </div>
  )
}

export default ModelCreateButton
