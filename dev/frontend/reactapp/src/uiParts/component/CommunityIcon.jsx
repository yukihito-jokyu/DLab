import React from 'react';
import '../css/CommunityIcon.css';
import { ReactComponent as Community } from '../../assets/svg/description_24.svg';
import { useNavigate, useParams } from 'react-router-dom';

function CommunityIcon() {
  const { projectName } = useParams()
  const navigate = useNavigate();
  const handleNave = () => {
    navigate(`/community/${projectName}`);
  }
  return (
    <div className='community-icon-wrapper' onClick={handleNave}>
      <Community className='community' />
    </div>
  )
}

export default CommunityIcon
