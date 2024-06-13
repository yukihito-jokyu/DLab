import React from 'react';
import './ImageClassificationProjectList.css';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';
import GradationFonts from '../../uiParts/component/GradationFonts';
import GradationButton from '../../uiParts/component/GradationButton';

function CreateField() {
  const fontStyle1 = {
    fontSize: '23px',
    fontWeight: '600',
    paddingTop: '35px'
  }
  const fontStyle2 = {
    fontSize: '23px',
    fontWeight: '600',
    paddingTop: '35px'
  }
  const fontStyle3 = {
    fontSize: '20px',
    fontWeight: '600'
  }
  return (
    <div>
      <div className='create-name-field'>
        <div className='create-name-wapper'>
          <div className='project-name'>
            {/* <p>Project Name</p> */}
            <GradationFonts text={'Project Name'} style={fontStyle1} />
          </div>
          <div className='project-name-field'>
            <p>Project Name</p>
          </div>
          <div>
          <  div className='projecttitle-line'></div>
          </div>
        </div>
      </div>
      <div className='create-dataset-field'>
        <div className='create-dataset-wrapper'>
          <div className='create-dataset-name'>
            <GradationFonts text={'Datasets'} style={fontStyle2} />
          </div>
          <div className='create-dataset-upload'>
            <GradationFonts text={'クリック・ドラッグアウトでアップロード'} style={fontStyle3} />
          </div>
          <div className='create-dataset-upload-name'>
            <GradationFonts text={'✔ [FileName.zip]'} style={fontStyle3} />
          </div>
        </div>
      </div>
      <div className='create-button-field'>
        <div className='create-button-position'>
        <div>
          <GradationButton text={'作成'} />
        </div>
        </div>
      </div>
      <div className='delet-button-field'>
        <div className='delet-button-wrapper'>
          <DeletIcon className='delet-svg' />
        </div>
      </div>
    </div>
  )
};

export default CreateField;
