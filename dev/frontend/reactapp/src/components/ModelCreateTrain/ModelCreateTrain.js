import React from 'react';
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import ProjectModelName from '../../uiParts/component/ProjectModelName';
import './ModelCreateTrain.css';
import ScreenField from './ScreenField';
import UserIcon from '../../uiParts/component/UserIcon';
import TrainButtons from '../../uiParts/component/TrainButtons';

function ModelCreateTrain() {
  return (
    <div className='mct-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
        projectmodelnamecomponent={ProjectModelName}
        usericoncomponent={UserIcon}
        trainbuttonscomponent={TrainButtons}
      />
      <ScreenField />
    </div>
  )
}

export default ModelCreateTrain
