import React from 'react';
import './ModelCreateBox.css';
import CreateName from '../ImageClassificationProjectList/CreateName';
import CreateButton from '../Login/CreateButton';
import CreateFieldDeletButton from '../ImageClassificationProjectList/CreateFieldDeletButton';

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
