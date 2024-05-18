import React from 'react';
import './LoginTitle.css';
import GradationFonts from '../../uiParts/component/GradationFonts';

function LoginTitle() {
  const text = 'login'
  const style = {
    fontSize: '30px'
  }
  return (
    <div className='login-title-wrapper'>
      <GradationFonts text={text} style={style} />
    </div>
  )
}

export default LoginTitle
