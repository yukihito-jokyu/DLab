import React from 'react';
import '../css/BurgerButton.css';

function BurgerButton({ handleClickMenu, menu }) {
  const className = menu ? 'bugerbutton-wrapper-open' : 'bugerbutton-wrapper'
  return (
    <div className={className} onClick={handleClickMenu} style={{ cursor: 'pointer' }}>
      <div className='buger-field'>
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  )
};

export default BurgerButton;