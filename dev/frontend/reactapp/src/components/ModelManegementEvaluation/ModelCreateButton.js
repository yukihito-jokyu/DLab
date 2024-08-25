import React from 'react';
import './ModelManegementEvaluation.css';
import { ReactComponent as AddIcon } from '../../assets/svg/project_add_48.svg'

function ModelCreateButton({ handleCreateModal }) {
  return (
    <div className='model-create-button-wrapper' onClick={handleCreateModal} style={{ cursor: 'pointer' }}>
      <AddIcon className='model-add-svg' />
    </div>
  )
}

export default ModelCreateButton
