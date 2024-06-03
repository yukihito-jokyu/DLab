import React from 'react';
import GradationButton from '../../uiParts/component/GradationButton';

function ProjectRecommend() {
  return (
    <div>
      <div className='classification-wrapper'>
        <div className='project-info-wrapper'>
          <p className='project-title'>Image Classification</p>
          <p className='project-title-ja'>画像分類</p>
          <p className='project-exp'>説明</p>
          <GradationButton />
        </div>
        <div></div>
      </div>
      <div className='reinforcement-wrapper'>
        <div className='project-info-wrapper'>
          <p className='project-title'>Reinforcement Learning</p>
          <p className='project-title-ja'>強化学習モデル</p>
          <p className='project-exp'>説明</p>
          <GradationButton />
        </div>
        <div></div>
        </div>
    </div>
  );
};

export default ProjectRecommend;
