import React from 'react'
import './AlertModal.css';
import GradationFonts from '../../uiParts/component/GradationFonts';
import GradationButton from '../../uiParts/component/GradationButton';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';

function AlertModal({ deleteModal, handleClick, sendText }) {
  const text = 'Attention';
  const text2 = 'OK';
  const style = {
    fontSize: "30px",
    fontWeight: "600"
  }
  return (
    <div>
      <div className='alert-modal-wrapper'></div>
      <div className='alert-modal-field-wrapper'>
        <div className='gradation-border'>
          <div className='gradation-wrapper'>
            <div className='alert-modal-field'>
              <div className='modal-title'>
                <GradationFonts text={text} style={style} />
              </div>
              <div className='gradation-border2-wrapper'>
                <div className='gradation-border2'></div>
              </div>
              <div className='alert-comment'>
                <p dangerouslySetInnerHTML={{ __html: sendText }} />
              </div>
              <div className='alert-modal'>
                <div onClick={handleClick}>
                  <GradationButton text={text2} />
                </div>
              </div>
              <div className='train-modal-delet-button-field' onClick={deleteModal} style={{ cursor: 'pointer' }}>
                <DeletIcon className='delet-svg' />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AlertModal;
