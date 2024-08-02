import React from 'react';
import '../css/CommunityIcon.css';
import { ReactComponent as Community } from '../../assets/svg/description_24.svg';
import { useNavigate } from 'react-router-dom';

function CommunityIcon() {
  const navigate = useNavigate();
  const handleNave = () => {
    navigate('/community');
  }
  return (
    <div className='community-icon-wrapper' onClick={handleNave}>
      <Community className='community' />
    </div>
  )
}

export default CommunityIcon
