import React from 'react';
import '../css/Logo.css';
import { ReactComponent as LogoSVG } from '../../assets/svg/DLAB-logo.svg';

function Logo() {
  return (
    <div className='logo-wrapper'>
      {/* <p>DLab</p> */}
      <LogoSVG className='logo-svg' />
    </div>
  )
}

export default Logo
