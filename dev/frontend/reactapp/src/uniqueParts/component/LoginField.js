import React from 'react';
import '../css/LoginField.css';
import LoginLogo from '../../uiParts/component/LoginLogo';
import LoginIdField from '../../uiParts/component/LoginIdField';
import CreateButton from '../../uiParts/component/CreateButton';
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
