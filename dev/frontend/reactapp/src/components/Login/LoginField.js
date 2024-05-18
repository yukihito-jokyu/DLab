import React from 'react';
import './LoginField.css';
import LoginLogo from './LoginLogo';
import LoginIdField from './LoginIdField';
import CreateButton from './CreateButton';
import GradationFonts from '../../uiParts/component/GradationFonts';

function LoginField() {
  const style = {
    fontSize: '10px'
  };
  return (
    <div>
      <LoginLogo />
      <LoginIdField />
      <LoginIdField />
      <div className='login-button-wrapper'>
        <CreateButton />
      </div>
      <div className='login-guest'>
        <GradationFonts text={'ゲストとしてログイン'} style={style} />
      </div>
    </div>
  )
}

export default LoginField
