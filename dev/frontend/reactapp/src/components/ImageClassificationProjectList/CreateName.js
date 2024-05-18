import React from 'react';
import './CreateName.css';
import ImageProjectTitleLine from './ImageProjectTitleLine';
import GradationFonts from '../../uiParts/component/GradationFonts';

function CreateName() {
  const text = 'Project Name';
  const style = {
    fontSize: '23px',
    fontWeight: '600',
    paddingTop: '35px'
  }
  return (
    <div className='create-name-wapper'>
      <div className='project-name'>
        {/* <p>Project Name</p> */}
        <GradationFonts text={text} style={style} />
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
