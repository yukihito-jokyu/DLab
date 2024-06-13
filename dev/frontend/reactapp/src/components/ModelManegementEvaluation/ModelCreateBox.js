import React from 'react';
import './ModelManegementEvaluation.css';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';
import GradationFonts from '../../uiParts/component/GradationFonts';
import GradationButton from '../../uiParts/component/GradationButton';

function ModelCreateBox() {
  const text1 = 'Model Name';
  const style = {
    fontSize: '23px',
    fontWeight: '600',
    paddingTop: '35px'
  }
  const text2 = '作成';
  return (
    <div className='model-create-box-border'>
      <div className='create-name-field'>
        <div className='create-name-wapper'>
          <div className='project-name'>
            {/* <p>Project Name</p> */}
            <GradationFonts text={text1} style={style} />
          </div>
          <div className='project-name-field'>
            <p>Project Name</p>
          </div>
          <div>
            <div className='projecttitle-line'>
            
            </div>
          </div>
        </div>
      </div>
      <div className='create-model-button-field'>
        <div className='create-model-button'>
          <div>
            <GradationButton text={text2} />
          </div>
        </div>
      </div>
      <div className='delet-button-field'>
        <div className='delet-button-wrapper'>
          <div className='delet-button-wrapper'>
            <DeletIcon className='delet-svg' />
          </div>
        </div>
      </div>
    </div>
  )
}

export default ModelCreateBox
