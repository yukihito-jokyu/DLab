import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../css/Logo.css';
import { ReactComponent as LogoSVG } from '../../assets/svg/DLAB-logo.svg';
import { ReactComponent as LogoSVG1 } from '../../assets/svg/DLAB-logo_type1.svg';
import { ReactComponent as LogoSVG5 } from '../../assets/svg/DLAB-logo_type5.svg';

function Logo() {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate('/top');
  };

  return (
    <div className='logo-wrapper'>
      {/* <p>DLab</p> */}
      {/* <LogoSVG className='logo-svg' /> */}
      <LogoSVG1 className='logo-svg' onClick={handleClick} style={{ cursor: 'pointer' }} />
      {/* <LogoSVG5 className='logo-svg' /> */}
    </div>
  );
}

export default Logo;
