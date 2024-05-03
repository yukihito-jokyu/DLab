import React from 'react';
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import '../css/ModelManegementEvaluation.css';
import ModelField from '../../uniqueParts/component/ModelField';
import ProjectModelName from '../../uiParts/component/ProjectModelName';
import ModelCreateField from '../../uniqueParts/component/ModelCreateField';

function ModelManegementEvaluation() {
  return (
    <div className='mme-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
        projectmodelnamecomponent={ProjectModelName}
      />
      <ModelField />
      <div className='create-background-field'>
        <ModelCreateField />
      </div>
    </div>
  )
}

export default ModelManegementEvaluation;
