import React from 'react';
import './Top.css'
import GradationButton from '../../uiParts/component/GradationButton';
import DLImage from '../../assets/images/DLImage.png';
import { useAuthState } from 'react-firebase-hooks/auth';
import { auth } from '../../db/firebase';
import { useNavigate } from 'react-router-dom';
import { signInWithGoogle } from '../../db/function/users';

function Explanation({ setFirstSignIn }) {
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
    <div className='explanation-wrapper'>
      <div className='exp-left'>
        <div>
          <p className='first-p'>DLab is<br />No code Deep Learning Tool !</p>
          <p className='second-p'>プログラミングの知識に影響されない良い学びにつなぐために<br />視覚的操作によるAIモデルのデザインツールを提案します</p>
        </div>
        <div className='button-wrapper-top' onClick={handleSignIn} style={{ cursor: 'pointer' }}>
          <GradationButton text={'launch'} />
        </div>
      </div>
      <div className='exp-right'>
        <img src={DLImage} alt='DLImage' className='DL-image' />
      </div>
    </div>
  );
};

export default Explanation;
