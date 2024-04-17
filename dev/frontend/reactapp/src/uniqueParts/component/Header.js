import React from 'react'
import '../css/Header.css'

function Header(props) {
  // 引数としてコンポーネントを受け取る
  const BurgerButtonComponent = props.burgerbutton;
  const LogoComponent = props.logocomponent;
  return (
    <div className='header-wrapper'>
      {BurgerButtonComponent && <BurgerButtonComponent />}
      {LogoComponent && <LogoComponent />}
    </div>
  )
}

export default Header;
