import React, { useState } from 'react';
import '../css/Header.css';
import Menu from '../../uiParts/component/Menu';

function Header(props) {
  const [menuOpen, setMenuOpen] = useState(false);
  const BurgerButtonComponent = props.burgerbutton;
  const LogoComponent = props.logocomponent;
  const ProjectModelNameComponent = props.projectmodelnamecomponent;
  const UserIconComponent = props.usericoncomponent;
  const TrainButtonsComponent = props.trainbuttonscomponent;
  const CommunityIconComponent = props.communityiconcomponent;
  // 引数として関数を受け取る
  const changeEdit = props.changeedit;
  const changeTrain = props.changeTrain;
  const changeVisImageModal = props.changeVisImageModal;
  const changeVisTrainModal = props.changeVisTrainModal;

  const handleClickMenu = () => {
    setMenuOpen(!menuOpen);
  };

  return (
    <div className='header-wrapper'>
      {BurgerButtonComponent && <BurgerButtonComponent handleClickMenu={handleClickMenu} menu={menuOpen} />}
      {LogoComponent && <LogoComponent />}
      {ProjectModelNameComponent && <ProjectModelNameComponent />}
      {UserIconComponent && <UserIconComponent />}
      {TrainButtonsComponent && <TrainButtonsComponent changeEdit={changeEdit} changeTrain={changeTrain} changeVisImageModal={changeVisImageModal} changeVisTrainModal={changeVisTrainModal} />}
      {CommunityIconComponent && <CommunityIconComponent />}
      <Menu handleClickMenu={handleClickMenu} menuOpen={menuOpen} />
    </div>
  );
}

export default Header;
