import React from 'react';
import '../css/CommunityIcon.css';
import { ReactComponent as Community } from '../../assets/svg/description_24.svg';
import { useNavigate, useParams } from 'react-router-dom';

function CommunityIcon() {
  const { projectName, task } = useParams()
  const navigate = useNavigate();
  const handleNave = () => {
    navigate(`/community/${task}/${projectName}`);
  }
  return (
    <div className='community-icon-wrapper' onClick={handleNave} style={{ cursor: 'pointer' }}>
      <Community className='community' />
    </div>
  )
}

export default CommunityIcon
