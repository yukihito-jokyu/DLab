import React, { useState } from 'react';
import './ModelManegementEvaluation.css';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';
import GradationFonts from '../../uiParts/component/GradationFonts';
import GradationButton from '../../uiParts/component/GradationButton';
import { setModel } from '../../db/firebaseFunction';

function ModelCreateBox({ handleCreateModal }) {
  const [modelName, setModelName] = useState("model_name");
  const text1 = 'Model Name';
  // モデルの作成
  const handleMakeModel = async () => {
    const userId = JSON.parse(sessionStorage.getItem('userId'));
    const projectId = JSON.parse(sessionStorage.getItem('projectId'));
    await setModel(userId, projectId, modelName);
  //   const response = await fetch('http://127.0.0.1:5000/train', {
  //     method: 'POST',
  //     headers: {
  //       'Content-Type': 'application/json',
  //     },
  //     body: JSON.stringify(AllData),
  //   });
    handleCreateModal();
  };
  const style = {
    fontSize: '23px',
    fontWeight: '600',
    paddingTop: '35px'
  };
  const text2 = '作成';
  const handleChange = (e) => {
    setModelName(e.target.value);
  };
  return (
    <div className='model-create-box-border'>
      <div className='create-name-field'>
        <div className='create-name-wapper'>
          <div className='project-name'>
            {/* <p>Project Name</p> */}
            <GradationFonts text={text1} style={style} />
          </div>
          <div className='project-name-field'>
            {/* <p>Project Name</p> */}
            <input type='text' value={modelName} onChange={handleChange} className='model-name-input' />
          </div>
          <div>
            <div className='projecttitle-line'>
            
            </div>
          </div>
        </div>
      </div>
      <div className='create-model-button-field'>
        <div className='create-model-button'>
          <div onClick={handleMakeModel}>
            <GradationButton text={text2} />
          </div>
        </div>
      </div>
      <div className='delet-button-field'>
        <div className='delet-button-wrapper'>
          <div className='delet-button-wrapper' onClick={handleCreateModal}>
            <DeletIcon className='delet-svg' />
          </div>
        </div>
      </div>
    </div>
  )
}

export default ModelCreateBox
