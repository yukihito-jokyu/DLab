import React from 'react';
import './CreateField.css';
import CreateName from './CreateName';
import CreateDataset from './CreateDataset';
import CreateButton from '../Login/CreateButton';
import CreateFieldDeletButton from './CreateFieldDeletButton';

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
