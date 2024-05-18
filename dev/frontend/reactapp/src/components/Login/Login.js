import React from 'react';
import './Login.css';
import LoginTitle from './LoginTitle';
import LoginField from './LoginField';
import BorderGradationBox from '../../uiParts/component/BorderGradationBox';

function Login() {
  const style1 = {
    width: '100%',
    height: '430px'
  }
  return (
    <div className='login-wrapper'>
      <div className='login-out-box'>
        <LoginTitle />
        {/* <LoginField /> */}
        <BorderGradationBox children={LoginField} style1={style1} />
      </div>
    </div>
  )
}

export default Login
