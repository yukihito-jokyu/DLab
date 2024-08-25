import React, { useEffect, useState } from 'react'
import './Top.css';
import Explanation from './Explanation';
import Header from '../../uniqueParts/component/Header';
import BurgerButton from '../../uiParts/component/BurgerButton';
import Logo from '../../uiParts/component/Logo';
import ProjectRecommend from './ProjectRecommend';
import YoutubeRecommend from './YoutubeRecommend';
import UpDatesRecommend from './UpDatesRecommend';
import Footer from './Footer';
import UserNameModal from './UserNameModal';
import { auth } from '../../db/firebase';
import { useAuthState } from 'react-firebase-hooks/auth';
import { getUserId } from '../../db/function/users';
import UserIcon from '../../uiParts/component/UserIcon';
import AlertModal from '../utils/AlertModal';



function Top() {
  const [firstSignIn, setFirstSignIn] = useState(false);
  const [sameName, setSameName] = useState(false);
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
          usericoncomponent={UserIcon}
        />
      </div>
      <Explanation setFirstSignIn={setFirstSignIn} />
      <ProjectRecommend setFirstSignIn={setFirstSignIn} />
      <YoutubeRecommend />
      <UpDatesRecommend setFirstSignIn={setFirstSignIn} />
      <Footer />
      {firstSignIn && <UserNameModal setFirstSignIn={setFirstSignIn} setSameName={setSameName} />}
      {sameName && <AlertModal deleteModal={() => setSameName(false)} handleClick={() => setSameName(false)} sendText={'その名前は既に存在します'} />}
    </div>
  )
}

export default Top
