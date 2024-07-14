import React from 'react';
import '../css/CommunityIcon.css';
import { ReactComponent as Community } from '../../assets/svg/description_24.svg';

function CommunityIcon() {
  return (
    <div className='community-icon-wrapper'>
      <Community className='community' />
    </div>
  )
}

export default CommunityIcon
