import React, { useContext } from 'react';
import './Top.css'
import GradationButton from '../../uiParts/component/GradationButton';
import { UserInfoContext } from '../../App';
import { auth } from '../../db/firebase';
import { useAuthState } from 'react-firebase-hooks/auth';
import { useNavigate } from 'react-router-dom';
import { signInWithGoogle } from '../../db/function/users';

function UpDatesRecommend({ setFirstSignIn }) {
  // const { setUserId, setFirstSignIn } = useContext(UserInfoContext);
  const [user] = useAuthState(auth);
  const navigate = useNavigate();
  const handleSignIn = () => {
    if (user) {
      navigate('/testfirebase');
    } else {
      signInWithGoogle(setFirstSignIn);
    }
  };
  return (
    <div className='update-wrapper'>
      <div className='update-title'>
        <p>UPDATEs</p>
      </div>
      <div className='update-info'>
        <p>今後のサービス向上のために何かご要望等ございましたら</p>
        <p>下記フォームへご要望下さい</p>
      </div>
      <div className='update-button' onClick={handleSignIn}>
        <GradationButton />
      </div>
      <div className='update-list'>
        <div className='update-tile'></div>
        <div className='update-tile'></div>
        <div className='update-tile'></div>
        <div className='update-tile'></div>
      </div>
    </div>
  );
};

export default UpDatesRecommend;
