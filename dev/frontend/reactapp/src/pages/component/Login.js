import React from 'react';
import '../css/Login.css';
import LoginTitle from '../../uiParts/component/LoginTitle';
import LoginField from '../../uniqueParts/component/LoginField';

function Login() {
  return (
    <div className='login-wrapper'>
      <div className='login-out-box'>
        <LoginTitle />
        <LoginField />
      </div>
    </div>
  )
}

export default Login
