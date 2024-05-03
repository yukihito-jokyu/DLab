import React from 'react';
import '../css/CreateName.css';
import ImageProjectTitleLine from '../../uiParts/component/ImageProjectTitleLine';

function CreateName() {
  return (
    <div className='create-name-wapper'>
      <div className='project-name'>
        <p>Project Name</p>
      </div>
      <div className='project-name-field'>
        <p>Project Name</p>
      </div>
      <div>
        <ImageProjectTitleLine />
      </div>
    </div>
  )
}

export default CreateName
