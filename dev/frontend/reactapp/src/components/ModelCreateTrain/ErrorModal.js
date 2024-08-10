import React from 'react'
import './ModelCreateTrain.css'
import GradationFonts from '../../uiParts/component/GradationFonts'
import GradationButton from '../../uiParts/component/GradationButton'

function ErrorModal({ handleErrorModal, filedName, tileName }) {
  const style = {
    fontSize: "36px",
    fontWeight: "600",
    background: "linear-gradient(91.27deg, #B69EFF 0.37%, #FF54D9 99.56%)"
  }
  return (
    <div>
      <div className='error-modal-wrapper'></div>
      <div className='error-field-wrapper'>
        <div className='modal-title'>
          <p>不正な操作</p>
        </div>
        <div className='gradation-border2-wrapper'>
          <div className='gradation-border2'></div>
        </div>
        <div className='error-comment-wrapper'>
          <div className='error-comment-field'>
            <p>このエリアは<span>{filedName}</span>なので、<br />レイヤー：<span>{tileName}</span>を追加することができません。</p>
          </div>
        </div>
        <div className='error-model-button-field'>
          <div onClick={handleErrorModal} style={{ cursor: 'pointer' }}>
            <GradationButton text={'OK'} style1={style} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default ErrorModal
