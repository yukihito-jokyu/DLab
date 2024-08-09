import React from 'react';
import './Profile.css';
import Header from '../../uniqueParts/component/Header';
import BurgerButton from '../../uiParts/component/BurgerButton';
import Logo from '../../uiParts/component/Logo';
import ProfileField from './ProfileField';
import UserIcon from '../../uiParts/component/UserIcon';

function Profile() {
  return (
    <div className='profile-wrapper'>
      <div className='profile-header-wrapper'>
        <Header
          burgerbutton={BurgerButton}
          logocomponent={Logo}
          usericoncomponent={UserIcon}
        />
      </div>
      <div className='profile-area'>
        <ProfileField />
      </div>
    </div>
  )
}

export default Profile
