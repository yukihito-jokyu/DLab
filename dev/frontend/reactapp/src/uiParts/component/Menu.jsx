import React from 'react';
import '../css/Menu.css';
import { useNavigate } from 'react-router-dom';

function Menu({ handleClickMenu }) {
  const userId = JSON.parse(sessionStorage.getItem('userId'));
  const navigate = useNavigate();
  const navTop = () => {
    navigate('/top')
  }
  const navImage = () => {
    navigate(`/ImageClassificationProjectList/${userId}`)
  }
  const navReinforcement = () => {
    navigate('/RLProjectList')
  }
  const navProjectShare = () => {
    navigate('/projectshare')
  }
  const navProfile = () => {
    navigate(`/profile/${userId}`)
  }
  return (
    <div className='menu-wrapper'>
      <div className='menu-field-wrapper' onClick={handleClickMenu}></div>
      <div className='menu-field'>
        <div className='menu-list-wrapper'>
          <p onClick={navTop} style={{ cursor: 'pointer' }}>・Top</p>
          <p onClick={navImage} style={{ cursor: 'pointer' }}>・Image Classification</p>
          <p onClick={navReinforcement} style={{ cursor: 'pointer' }}>・Reinforcement Learning</p>
          <p onClick={navProjectShare} style={{ cursor: 'pointer' }}>・Projects Share</p>
          <p onClick={navProfile} style={{ cursor: 'pointer' }}>・Profile</p>
        </div>
      </div>
    </div>
  )
}

export default Menu;
