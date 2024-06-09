import React from 'react';
import './ProjectShare.css';
import ProjectShareImage from '../../assets/images/ProjectShareImage.png'
import GradationButton from '../../uiParts/component/GradationButton';

function ProjectRequest() {
  const style1 = {
    width: '200px',
    background: 'linear-gradient(95.34deg, #B6F862 3.35%, #00957A 100%), linear-gradient(94.22deg, #D997FF 0.86%, #50BCFF 105.96%)'
  };
  return (
    <div className='project-request-wrapper'>
      <div className='request-left'>
        <p className='title'>Projects</p>
        <p className='info-1'>DeepLearningに用いるデータセットを</p>
        <p className='info-2'>このページからインポートすることができます。</p>
        <div className='button-wrapper'>
          <div className='button'>
            <GradationButton text={'Request Project'} style1={style1} />
          </div>
          <p className='info-3'>▲ データセットをリクエストする</p>
        </div>
      </div>
      <div className='request-right'>
        <img src={ProjectShareImage} alt='ProjectShareImage' className='project-share-image' />
      </div>
    </div>
  );
};

export default ProjectRequest;
