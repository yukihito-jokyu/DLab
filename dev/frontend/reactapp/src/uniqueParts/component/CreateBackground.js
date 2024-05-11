import React from 'react';
import '../css/CreateBackground.css';
import CreateField from './CreateField';
import BorderGradationBox from '../../uiParts/component/BorderGradationBox';

function CreateBackground() {
  const style1 = {
    width: '906px',
    height: '506px'
  }
  const style2 = {
    position: 'relative',
    padding: '0px'
  }
  return (
    <div className='create-background-wrapper'>
      <div className='create-background-color'></div>
      <BorderGradationBox children={CreateField} style1={style1} style2={style2} />
    </div>
  )
};

export default CreateBackground;
