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
          <p onClick={navTop}>・Top</p>
          <p onClick={navImage}>・Image Classification</p>
          <p onClick={navReinforcement}>・Reinforcement Learning</p>
          <p onClick={navProjectShare}>・Projects Share</p>
          <p onClick={navProfile}>・Profile</p>
        </div>
      </div>
    </div>
  )
}

export default Menu;
