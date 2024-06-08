import React, { useContext, useState } from 'react'
import './Top.css';
import Explanation from './Explanation';
import Header from '../../uniqueParts/component/Header';
import BurgerButton from '../../uiParts/component/BurgerButton';
import Logo from '../../uiParts/component/Logo';
import Recommend from './Recommend';
import ProjectRecommend from './ProjectRecommend';
import YoutubeRecommend from './YoutubeRecommend';
import UpDatesRecommend from './UpDatesRecommend';
import Footer from './Footer';
import UserNameModal from './UserNameModal';
// import { useAuthState } from 'react-firebase-hooks/auth';
// import { auth } from '../../db/firebase';
import { UserInfoContext } from '../../App';



function Top() {
  const { firstSignIn } = useContext(UserInfoContext);
  // const [user] = useAuthState(auth);
  // console.log(userId)
  return (
    <div className='top-wrapper'>
      <div className='top-header-wrapper'>
        <Header 
          burgerbutton={BurgerButton}
          logocomponent={Logo}
        />
      </div>
      <Explanation />
      <Recommend />
      <ProjectRecommend />
      <YoutubeRecommend />
      <UpDatesRecommend />
      <Footer />
      {firstSignIn && <UserNameModal />}
    </div>
  )
}

export default Top
