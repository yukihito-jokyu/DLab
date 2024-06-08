import React from 'react';
import './RLProjectList.css';
import GradationButton from '../../uiParts/component/GradationButton';

function RLRequestIcon() {
  const style1 = {
    background: '#96AAFF'
  }
  return (
    <div className='rl-request-icon-wrapper'>
      <div className='request-wrapper'>
        <p className='request'>Comming soon ...</p>
        <div className='button-wrapper'>
          <GradationButton text={'Request'} style1={style1} />
        </div>
      </div>
    </div>
  )
}

export default RLRequestIcon
