import React from 'react';
import '../css/Login.css';
import LoginTitle from '../../uiParts/component/LoginTitle';
import LoginField from '../../uniqueParts/component/LoginField';
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
