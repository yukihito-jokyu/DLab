import React from 'react'
import './ImageProjectField.css'
import ImageProjectIcon from './ImageProjectIcon'
import ImageProjectAdd from './ImageProjectAdd'
import BorderGradationBox from '../../uiParts/component/BorderGradationBox'


function ImageProjectField() {
  const style1 = {
    margin: '0px 40px 40px 0',
    width: '276px',
    height: '241px'
  }
  return (
    <div className='imageprojectfield-wrapper'>
      <BorderGradationBox children={ImageProjectIcon} style1={style1} />
      <ImageProjectAdd />
    </div>
  )
}

export default ImageProjectField
