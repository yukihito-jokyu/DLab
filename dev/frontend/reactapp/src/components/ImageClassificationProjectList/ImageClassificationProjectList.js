import React, { useEffect } from 'react'
import ImageProjectField from './ImageProjectField'
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import './ImageClassificationProjectList.css'
import ContentsBackground from '../../uiParts/component/ContentsBackground'
import { useNavigate } from 'react-router-dom'
import UserIcon from '../../uiParts/component/UserIcon'

function ImageClassificationProjectList() {
  const navigate = useNavigate();
  useEffect(() => {
    const fatchProjects = () => {
      const userId = JSON.parse(sessionStorage.getItem('userId'));
      if (userId === null) {
        navigate('/top');
      };
    };

    fatchProjects();

  }, [navigate]);
  return (
    <div className='projectlist-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
        usericoncomponent={UserIcon}
      />
      <ContentsBackground children={ImageProjectField} />
    </div>
  )
}

export default ImageClassificationProjectList;
