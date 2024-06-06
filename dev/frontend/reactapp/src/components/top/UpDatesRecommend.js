import React from 'react';
import './Top.css'
import GradationButton from '../../uiParts/component/GradationButton';

function UpDatesRecommend() {
  return (
    <div className='update-wrapper'>
      <div className='update-title'>
        <p>UPDATEs</p>
      </div>
      <div className='update-info'>
        <p>今後のサービス向上のために何かご要望等ございましたら</p>
        <p>下記フォームへご要望下さい</p>
      </div>
      <div className='update-button'>
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
