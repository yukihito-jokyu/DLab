import React from 'react';
import './EditScreen.css';
import InputField from './InputField';
import OutputField from './OutputField';
import MiddleFeild from './MiddleFeild';

function EditScreen() {
  return (
    <div className='edit-screen-wrapper'>
      <InputField />
      <MiddleFeild />
      <OutputField />
    </div>
  )
}

export default EditScreen
