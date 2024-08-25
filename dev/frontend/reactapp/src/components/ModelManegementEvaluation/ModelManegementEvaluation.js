import React, { useEffect } from 'react';
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import './ModelManegementEvaluation.css';
import ModelField from './ModelField';
import ProjectModelName from '../../uiParts/component/ProjectModelName';
import ContentsBackground from '../../uiParts/component/ContentsBackground';
import { useNavigate } from 'react-router-dom';
import CommunityIcon from '../../uiParts/component/CommunityIcon';
import UserIcon from '../../uiParts/component/UserIcon';

function ModelManegementEvaluation() {
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
    <div className='mme-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
        projectmodelnamecomponent={ProjectModelName}
        communityiconcomponent={CommunityIcon}
        usericoncomponent={UserIcon}
      />
      <ContentsBackground children={ModelField} />
    </div>
  )
}

export default ModelManegementEvaluation;
