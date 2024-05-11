import React from 'react';
import '../css/ModelCreateBox.css';
import CreateName from './CreateName';
import CreateButton from '../../uiParts/component/CreateButton';
import CreateFieldDeletButton from '../../uiParts/component/CreateFieldDeletButton';

function ModelCreateBox() {
  return (
    <div className='model-create-box-border'>
      <div className='create-name-field'>
        <CreateName />
      </div>
      <div className='create-model-button-field'>
        <div className='create-model-button'>
          <CreateButton />
        </div>
      </div>
      <div className='delet-button-field'>
        <CreateFieldDeletButton />
      </div>
    </div>
  )
}

export default ModelCreateBox
