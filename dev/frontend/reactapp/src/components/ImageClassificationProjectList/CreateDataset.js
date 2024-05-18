import React from 'react';
import './CreateDataset.css';
import GradationFonts from '../../uiParts/component/GradationFonts';

function CreateDataset() {
  const style1 = {
    fontSize: '23px',
    fontWeight: '600',
    paddingTop: '35px'
  }
  const style2 = {
    fontSize: '20px',
    fontWeight: '600'
  }
  return (
    <div className='create-dataset-wrapper'>
      <div className='create-dataset-name'>
        <GradationFonts text={'Datasets'} style={style1} />
      </div>
      <div className='create-dataset-upload'>
        <GradationFonts text={'クリック・ドラッグアウトでアップロード'} style={style2} />
      </div>
      <div className='create-dataset-upload-name'>
        <GradationFonts text={'✔ [FileName.zip]'} style={style2} />
      </div>
    </div>
  )
}

export default CreateDataset
