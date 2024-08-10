import React, { useEffect, useState } from 'react';
import '../css/UserIcon.css'
import { useNavigate } from 'react-router-dom';
import { getImage } from '../../db/function/storage';

function UserIcon() {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const [userImage, setUserImage] = useState(null);
  const navigate = useNavigate();
  // ユーザー画像
  useEffect(() => {
    const fetchUserImage = async () => {
      const url = await getImage(`images/${userId}`);
      setUserImage(url)
    }
    fetchUserImage();
  }, [userId]);
  const handleNav = () => {
    navigate(`/profile/${userId}`);
  }
  return (
    <div className='user-icon-wrapper' onClick={handleNav} style={{ cursor: 'pointer' }}>
      <div className='user-icon'>
        <img src={userImage} alt='icon' />
      </div>
    </div>
  )
}

export default UserIcon
