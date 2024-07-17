import React, { useState } from 'react';
import Header from '../../uniqueParts/component/Header'
import BurgerButton from '../../uiParts/component/BurgerButton'
import Logo from '../../uiParts/component/Logo'
import ProjectModelName from '../../uiParts/component/ProjectModelName';
import './ModelCreateTrain.css';
import ScreenField from './ScreenField';
import UserIcon from '../../uiParts/component/UserIcon';
import TrainButtons from '../../uiParts/component/TrainButtons';

function ModelCreateTrain() {
  const [edit, setEdit] = useState(true);
  const [train, setTrain] = useState(false);
  const changeEdit = () => {
    setEdit(!edit);
  };
  const changeTrain = () => {
    setTrain(!train);
  }
  return (
    <div className='mct-wrapper'>
      <Header
        burgerbutton={BurgerButton}
        logocomponent={Logo}
        projectmodelnamecomponent={ProjectModelName}
        usericoncomponent={UserIcon}
        trainbuttonscomponent={TrainButtons}
        changeedit={changeEdit}
        changeTrain={changeTrain}
      />
      <ScreenField edit={edit} train={train} changeTrain={changeTrain} />
    </div>
  )
}

export default ModelCreateTrain
