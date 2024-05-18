import React from 'react';
import './LoginIdField.css';
import GradationFonts from '../../uiParts/component/GradationFonts';

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
