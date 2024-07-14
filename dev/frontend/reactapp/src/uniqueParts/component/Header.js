import React, { useState } from 'react'
import '../css/Header.css'
import Menu from '../../uiParts/component/Menu';

function Header(props) {
  const [menu, setMenu] = useState(false);
  // 引数としてコンポーネントを受け取る
  const BurgerButtonComponent = props.burgerbutton;
  const LogoComponent = props.logocomponent;
  const ProjectModelNameComponent = props.projectmodelnamecomponent;
  const UserIconComponent = props.usericoncomponent;
  const TrainButtonsComponent = props.trainbuttonscomponent;
  const CommunityIconComponent = props.communityiconcomponent;
  const handleClickMenu = () => {
    setMenu(!menu);
  };
  return (
    <div className='header-wrapper'>
      {BurgerButtonComponent && <BurgerButtonComponent handleClickMenu={handleClickMenu} menu={menu} />}
      {LogoComponent && <LogoComponent />}
      {ProjectModelNameComponent && <ProjectModelNameComponent />}
      {UserIconComponent && <UserIconComponent />}
      {TrainButtonsComponent && <TrainButtonsComponent />}
      {CommunityIconComponent && <CommunityIconComponent />}
      {menu && <Menu />}
    </div>
  )
}

export default Header;
