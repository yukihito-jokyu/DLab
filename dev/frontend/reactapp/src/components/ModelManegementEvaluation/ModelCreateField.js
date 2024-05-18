import React from 'react';
import './ModelCreateField.css';
import ModelCreateBox from './ModelCreateBox';
import BorderGradationBox from '../../uiParts/component/BorderGradationBox';

function ModelCreateField() {
  const style1 = {
    width: '906px',
    height: '306px'
  }
  const style2 = {
    position: 'relative',
    padding: '0px'
  }
  return (
    <div className='create-background-wrapper'>
      <div className='create-background-color'></div>
      <BorderGradationBox children={ModelCreateBox} style1={style1} style2={style2} />
    </div>
  )
}

export default ModelCreateField
