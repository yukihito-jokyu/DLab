import React, { useEffect } from 'react';
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import './ModelManegementEvaluation.css';
import ModelField from './ModelField';
import ProjectModelName from '../../uiParts/component/ProjectModelName';
import ContentsBackground from '../../uiParts/component/ContentsBackground';
import ModelCreateField from './ModelCreateField';
import { useNavigate } from 'react-router-dom';
import CommunityIcon from '../../uiParts/component/CommunityIcon';

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
      />
      {/* <ModelField /> */}
      <ContentsBackground children={ModelField} />
      
    </div>
  )
}

export default ModelManegementEvaluation;
