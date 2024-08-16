import React from 'react';
import '../css/Menu.css';
import { useNavigate, useLocation } from 'react-router-dom';
import { ReactComponent as HomeIcon } from '../../assets/svg/home.svg';
import { ReactComponent as ImageIcon } from '../../assets/svg/image.svg';
import { ReactComponent as BirdIcon } from '../../assets/svg/bird.svg';
import { ReactComponent as ShereIcon } from '../../assets/svg/shere.svg';
import { ReactComponent as ProfileIcon } from '../../assets/svg/profile.svg';

function Menu({ handleClickMenu, menuOpen }) {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const navigate = useNavigate();
  const location = useLocation();

  const handleNavigation = (path) => {
    if (location.pathname === path) {
      handleClickMenu();
    } else {
      navigate(path);
      handleClickMenu();
    }
  };

  const navTop = () => handleNavigation('/top');
  const navImage = () => handleNavigation(`/ImageClassificationProjectList/${userId}`);
  const navReinforcement = () => handleNavigation('/RLProjectList');
  const navProjectShare = () => handleNavigation('/projectshare');
  const navProfile = () => handleNavigation(`/profile/${userId}`);

  return (
    <>
      {menuOpen && (
        <div className='menu-field-wrapper' onClick={handleClickMenu}></div>
      )}
      <div className={`menu-field ${menuOpen ? '' : 'hidden'}`}>
        <div className='menu-field-inner'>
          <div className='menu-list-wrapper'>
            <div onClick={navTop} style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
              <HomeIcon style={{ marginRight: '15px', width: '30px', height: '30px' }} />
              <p style={{ margin: 15, lineHeight: '30px' }}>ホーム</p>
            </div>
            <div onClick={navImage} style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
              <ImageIcon style={{ marginRight: '15px', width: '30px', height: '30px' }} />
              <p style={{ margin: 15, lineHeight: '30px' }}>画像分類</p>
            </div>
            <div onClick={navReinforcement} style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
              <BirdIcon style={{ marginRight: '15px', width: '30px', height: '30px' }} />
              <p style={{ margin: 15, lineHeight: '30px' }}>強化学習</p>
            </div>
            <div onClick={navProjectShare} style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
              <ShereIcon style={{ marginRight: '15px', width: '30px', height: '30px' }} />
              <p style={{ margin: 15, lineHeight: '30px' }}>プロジェクト一覧</p>
            </div>
            <div onClick={navProfile} style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
              <ProfileIcon style={{ marginRight: '15px', width: '30px', height: '30px' }} />
              <p style={{ margin: 15, lineHeight: '30px' }}>プロフィール</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default Menu;
