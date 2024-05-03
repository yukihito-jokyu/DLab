import React from 'react';
import '../css/EditScreen.css';
import InputField from './InputField';
import OutputField from './OutputField';

function EditScreen() {
  return (
    <div className='edit-screen-wrapper'>
      <InputField />
      <OutputField />
    </div>
  )
}

export default EditScreen
