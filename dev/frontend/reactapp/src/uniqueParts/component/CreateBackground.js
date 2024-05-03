import React from 'react';
import '../css/CreateBackground.css';
import CreateField from './CreateField';

function CreateBackground() {
  return (
    <div className='create-background-wrapper'>
      <div className='create-background-color'></div>
      <CreateField />
    </div>
  )
};

export default CreateBackground;
