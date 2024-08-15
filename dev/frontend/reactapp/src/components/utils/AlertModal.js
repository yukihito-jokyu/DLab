import React, { useEffect, useState } from 'react';
import './AlertModal.css';
import GradationFonts from '../../uiParts/component/GradationFonts';
import GradationButton from '../../uiParts/component/GradationButton';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';

function AlertModal({ deleteModal, handleClick, sendText }) {
  const [showModal, setShowModal] = useState(false);
  const [visible, setVisible] = useState(true);
  const [hideBackground, setHideBackground] = useState(false);
  const text = 'Attention';
  const text2 = 'OK';
  const style = {
    fontSize: "30px",
    fontWeight: "600"
  }

  useEffect(() => {
    setShowModal(true);
  }, []);

  const handleClose = () => {
    setHideBackground(true);
    setShowModal(false);
    setTimeout(() => {
      setVisible(false);
      deleteModal();
    }, 500);
  };

  return (
    visible && (
      <div className={`modal-container ${showModal ? 'show' : ''}`}>
        <div className={`alert-modal-wrapper ${hideBackground ? 'hide' : ''}`}></div>
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
                  <div onClick={() => { handleClick(); handleClose(); }}>
                    <GradationButton text={text2} />
                  </div>
                </div>
                <div className='train-modal-delet-button-field' onClick={() => { handleClick(); handleClose(); }} style={{ cursor: 'pointer' }}>
                  <DeletIcon className='delet-svg' />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  );
}


export default AlertModal;
