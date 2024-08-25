import React from 'react';
import './Top.css'
import GradationButton from '../../uiParts/component/GradationButton';

function UpDatesRecommend({ setFirstSignIn }) {
  return (
    <div className='update-wrapper'>
      <div className='update-title'>
        <p>UPDATEs</p>
      </div>
      <div className='update-info'>
        <p>今後のサービス向上のために何かご要望等ございましたら</p>
        <p>下記フォームへご要望下さい</p>
      </div>
      <div className='update-button' style={{ cursor: 'pointer' }}>
        <GradationButton text={'form'} />
      </div>
      <div className='update-list'>
      </div>
    </div>
  );
};

export default UpDatesRecommend;
