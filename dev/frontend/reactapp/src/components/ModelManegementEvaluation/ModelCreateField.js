import React from 'react';
import ModelCreateBox from './ModelCreateBox';
import BorderGradationBox from '../../uiParts/component/BorderGradationBox';

function ModelCreateField({ handleCreateModal }) {
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
      <BorderGradationBox style1={style1} style2={style2} >
        <ModelCreateBox handleCreateModal={handleCreateModal} />
      </BorderGradationBox>
    </div>
  )
}

export default ModelCreateField
