import React from 'react';
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import './ModelManegementEvaluation.css';
import ModelField from './ModelField';
import ProjectModelName from '../../uiParts/component/ProjectModelName';
import ModelCreateField from './ModelCreateField';
import ContentsBackground from '../../uiParts/component/ContentsBackground';

function ModelManegementEvaluation() {
  return (
    <div className='mme-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
        projectmodelnamecomponent={ProjectModelName}
      />
      {/* <ModelField /> */}
      <ContentsBackground children={ModelField} />
      <div className='create-background-field'>
        <ModelCreateField />
      </div>
    </div>
  )
}

export default ModelManegementEvaluation;
