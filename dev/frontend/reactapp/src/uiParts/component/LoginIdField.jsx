import React from 'react';
import '../css/LoginIdField.css';
import GradationFonts from './GradationFonts';

function LoginIdField() {
  const text = 'ID'
  return (
    <div className='login-id-wrapper'>
      <GradationFonts text={text} />
      <div className='id-input'></div>
    </div>
  )
}

export default LoginIdField
