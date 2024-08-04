import React, { useContext, useEffect, useState } from 'react'
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
import { UserInfoContext } from '../../App';
import { auth } from '../../db/firebase';
import { useAuthState } from 'react-firebase-hooks/auth';
import { getUserId } from '../../db/function/users';



function Top() {
  // const { firstSignIn } = useContext(UserInfoContext);
  const [firstSignIn, setFirstSignIn] = useState(false);
  const [user] = useAuthState(auth);
  
  // サインインしていた時、firebaseからuserIdを取得する
  useEffect(() => {
    const fetchUserId = () => {
      if (user) {
        getUserId(auth.currentUser.email);
      };
    };
    fetchUserId();
  }, [user]);
  
  return (
    <div className='top-wrapper'>
      <div className='top-header-wrapper'>
        <Header 
          burgerbutton={BurgerButton}
          logocomponent={Logo}
        />
      </div>
      <Explanation setFirstSignIn={setFirstSignIn} />
      <Recommend />
      <ProjectRecommend setFirstSignIn={setFirstSignIn} />
      <YoutubeRecommend />
      <UpDatesRecommend setFirstSignIn={setFirstSignIn} />
      <Footer />
      {firstSignIn && <UserNameModal setFirstSignIn={setFirstSignIn} />}
    </div>
  )
}

export default Top
