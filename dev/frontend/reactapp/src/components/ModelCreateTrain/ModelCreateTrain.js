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
  const [visImageModal, setVisImageModal] = useState(false);
  const [visTrainModal, setVisTrainModal] = useState(false);
  const changeEdit = () => {
    setEdit(!edit);
  };
  const changeTrain = () => {
    setTrain(!train);
  }
  const changeVisImageModal = () => {
    setVisImageModal(!visImageModal);
  }
  const changeVisTrainModal = () => {
    setVisTrainModal(!visTrainModal);
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
        changeVisImageModal={changeVisImageModal}
        changeVisTrainModal={changeVisTrainModal}
      />
      <ScreenField edit={edit} train={train} visImageModal={visImageModal} visTrainModal={visTrainModal} changeTrain={changeTrain} changeVisImageModal={changeVisImageModal} changeVisTrainModal={changeVisTrainModal} />
    </div>
  )
}

export default ModelCreateTrain
