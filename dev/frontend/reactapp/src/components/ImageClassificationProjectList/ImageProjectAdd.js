import React from 'react';
import './ImageClassificationProjectList.css';
import { ReactComponent as AddIcon } from '../../assets/svg/project_add_48.svg'

function ImageProjectAdd() {
  return (
    <div className='ImageProjectAdd-wrapper'>
      <AddIcon className='add-svg' />
    </div>
  )
};

export default ImageProjectAdd;
