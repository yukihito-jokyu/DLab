import React from 'react';
import './ModelManegementEvaluation.css';
import { ReactComponent as DLButtonIcon } from '../../assets/svg/download_40.svg'

function DLButton() {
  return (
    <div className='DL-button-wrapper'>
      <DLButtonIcon className='DL-button-icon' />
      <p>DL</p>
    </div>
  )
}

export default DLButton
