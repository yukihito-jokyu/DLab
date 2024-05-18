import React from 'react';
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import ProjectModelName from '../../uiParts/component/ProjectModelName';
import './ModelCreateTrain.css';
import ScreenField from './ScreenField';

function ModelCreateTrain() {
  return (
    <div className='mct-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
        projectmodelnamecomponent={ProjectModelName}
      />
      <ScreenField />
    </div>
  )
}

export default ModelCreateTrain
