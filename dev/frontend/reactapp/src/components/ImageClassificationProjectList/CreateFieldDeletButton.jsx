import React from 'react';
import './CreateFieldDeletButton.css';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';

function CreateFieldDeletButton() {
  return (
    <div className='delet-button-wrapper'>
      <DeletIcon className='delet-svg' />
    </div>
  )
}

export default CreateFieldDeletButton
