import React from 'react';
import '../css/CreateField.css';
import CreateName from './CreateName';
import CreateDataset from './CreateDataset';
import CreateButton from '../../uiParts/component/CreateButton';
import CreateFieldDeletButton from '../../uiParts/component/CreateFieldDeletButton';

function CreateField() {
  return (
    <div>
      <div className='create-name-field'>
        <CreateName />
      </div>
      <div className='create-dataset-field'>
        <CreateDataset />
      </div>
      <div className='create-button-field'>
        <div className='create-button-position'>
          <CreateButton />
        </div>
      </div>
      <div className='delet-button-field'>
        <CreateFieldDeletButton />
      </div>
    </div>
  )
};

export default CreateField;
