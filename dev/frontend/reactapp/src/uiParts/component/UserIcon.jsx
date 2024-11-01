import React, { useEffect, useState } from 'react';
import '../css/UserIcon.css';
import { useNavigate } from 'react-router-dom';
import { getImage } from '../../db/function/storage';
import DefoIcon from '../../assets/images/DLab_logo_400.png';

function UserIcon() {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const [userImage, setUserImage] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserImage = async () => {
      const storedImage = sessionStorage.getItem(`userImage_${userId}`);
      if (storedImage) {
        setUserImage(storedImage);
      } else {
        const url = await getImage(`images/${userId}`);
        if (url) {
          setUserImage(url);
          sessionStorage.setItem(`userImage_${userId}`, url);
        } else {
          console.log('No image found or failed to fetch image.');
        }
      }
    };

    if (userId) {
      fetchUserImage();
    }
  }, [userId]);

  const handleNav = () => {
    navigate(`/profile/${userId}`);
  };

  return (
    <div className='user-icon-wrapper' onClick={handleNav} style={{ cursor: 'pointer' }}>
      <div className={`user-icon ${!userImage ? 'default-icon-background' : ''}`}>
        {userImage ? (
          <img src={userImage} alt='icon' />
        ) : (
          <img src={DefoIcon} alt='default icon' />
        )}
      </div>
    </div>
  );
}

export default UserIcon;
