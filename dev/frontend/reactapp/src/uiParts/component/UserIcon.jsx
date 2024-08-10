import React from 'react';
import '../css/UserIcon.css'
import { useNavigate } from 'react-router-dom';

function UserIcon() {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const navigate = useNavigate();
  const handleNav = () => {
    navigate(`/profile/${userId}`);
  }
  return (
    <div className='user-icon-wrapper' onClick={handleNav} style={{ cursor: 'pointer' }}>
      <div className='user-icon'></div>
    </div>
  )
}

export default UserIcon
