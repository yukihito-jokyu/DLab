import React from 'react';
import '../css/LoginField.css';
import LoginLogo from '../../uiParts/component/LoginLogo';
import LoginIdField from '../../uiParts/component/LoginIdField';
import CreateButton from '../../uiParts/component/CreateButton';

function LoginField() {
  return (
    <div className='login-field-border'>
      <div className='login-field-wrapper'>
        <LoginLogo />
        <LoginIdField />
        <LoginIdField />
        <div className='login-button-wrapper'>
          <CreateButton />
        </div>
      </div>
    </div>
  )
}

export default LoginField
