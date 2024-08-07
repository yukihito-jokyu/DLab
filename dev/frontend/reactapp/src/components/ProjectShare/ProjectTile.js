import React, { useContext } from 'react';
import './ProjectShare.css';
import CIFAR10Image from '../../assets/images/CIFA10Image.png';
import { UserInfoContext } from '../../App';
import { useNavigate } from 'react-router-dom';

function ProjectTile({ title, info, style1, style2 }) {
  const { setProjectId } = useContext(UserInfoContext);
  const navigate = useNavigate();
  const handleNav = () => {
    setProjectId(title);
    sessionStorage.setItem('projectId', JSON.stringify(title));
    navigate(`/community/${title}`);
  }
  
  return (
    <div className='project-tile-wrapper' onClick={handleNav}>
      <div className='tile-wrapper'>
        <div className='tile-left' style={style1}>
          <p className='project-tile-title'>{title}</p>
          <p dangerouslySetInnerHTML={{ __html: info }} className='project-tile-info'></p>
        </div>
        <div className='tile-right'>
          <img src={CIFAR10Image} alt='CIFAR10Image' className='CIFAR-10-image' />
          <div className='image-cover' style={style2}></div>
        </div>
      </div>
    </div>
  );
};

export default ProjectTile
