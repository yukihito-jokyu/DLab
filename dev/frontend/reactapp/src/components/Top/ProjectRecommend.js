import React, { useContext, useEffect, useState } from 'react';
import './Top.css'
import GradationButton from '../../uiParts/component/GradationButton';
import { UserInfoContext } from '../../App';
import { auth } from '../../db/firebase';
import { useAuthState } from 'react-firebase-hooks/auth';
import { useNavigate } from 'react-router-dom';
import { signInWithGoogle } from '../../db/function/users';
import ICImage from '../../assets/images/IC_image1.jpg';
import RLImage from '../../assets/images/RL_image1.jpg';

function ProjectRecommend({ setFirstSignIn }) {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const [height, setHeight] = useState(0);
  useEffect(() => {
    const calculateHeight = () => {
      const element = document.getElementById('project-image-width');
      const width = element.offsetWidth;
      setHeight(width);
    };
    // 初期計算
    calculateHeight();
    window.addEventListener('resize', calculateHeight);
    return () => window.removeEventListener('resize', calculateHeight);
  }, []);

  const imageStyle = {
    height: `${height}px`
  }
  const imageClassificationStyle = {
    background: '#D997FF'
  }
  const reinforcementStyle = {
    background: '#50BBFF'
  }

  // const { setUserId, setFirstSignIn } = useContext(UserInfoContext);
  const [user] = useAuthState(auth);
  const navigate = useNavigate();
  const handleICSignIn = () => {
    if (user) {
      navigate(`/ImageClassificationProjectList/${userId}`);
    } else {
      signInWithGoogle(setFirstSignIn);

    }
  };
  const handleRLSignIn = () => {
    if (user) {
      navigate(`/RLProjectList`);
    } else {
      signInWithGoogle(setFirstSignIn);
    }
  };
  return (
    <div className='project-recommend-wrapper'>
      <div className='project-recommend-title'>
        <p>利用できる深層学習</p>
      </div>
      <div className='classification-wrapper'>
        <div className='project-info-wrapper'>
          <p className='project-title'>Image Classification</p>
          <p className='project-title-ja'>画像分類</p>
          <p className='project-exp'>画像分類とは、画像に含まれる物体を特定のカテゴリーやクラスに分類するタスクです。<br/>DLabでは、「MNIST」「FashionMNIST」「CIFAR10」「CIFAR100」の画像分類を体験することができます。<br/>高い精度を目指して頑張ってください！</p>
          <div className='project-button' onClick={handleICSignIn} style={{ cursor: 'pointer' }}>
            <GradationButton text={'画像分類へ'} style1={imageClassificationStyle} />
          </div>
        </div>
        <div className='project-image' id='project-image-width' style={imageStyle}>
          <img src={ICImage} alt='' />
        </div>
      </div>
      <div className='reinforcement-wrapper'>
        <div className='project-image' style={imageStyle}>
          <img src={RLImage} alt='' />
        </div>
        <div className='project-info-wrapper'>
          <p className='project-title'>Reinforcement Learning</p>
          <p className='project-title-ja'>強化学習モデル</p>
          <p className='project-exp'>強化学習とは、AIが環境と相互作用を通じて、行動の選択しを学ぶ手法です。<br/>DLabでは、「CartPole」「FlappyBird」の強化学習を体験することができます。</p>
          <div className='project-button' onClick={handleRLSignIn} style={{ cursor: 'pointer' }}>
            <GradationButton text={'強化学習へ'} style1={reinforcementStyle} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProjectRecommend;
