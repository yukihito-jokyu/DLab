import React, { useState, useEffect } from 'react';
import './ModelCreateTrain.css';
import GradationButton from '../../uiParts/component/GradationButton';

function ErrorModal({ handleErrorModal, filedName, tileName }) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const style = {
    fontSize: "36px",
    fontWeight: "600",
    background: "linear-gradient(91.27deg, #B69EFF 0.37%, #FF54D9 99.56%)"
  };

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      handleErrorModal();
    }, 200);
  };

  const modalStyle = {
    opacity: isVisible ? 1 : 0,
    transition: 'opacity 0.2s ease, transform 0.2s ease',
  };

  const backgroundStyle = {
    opacity: isVisible ? 0.7 : 0,
    transition: 'opacity 0.2s ease',
  };

  return (
    <div>
      <div className='error-modal-wrapper' style={backgroundStyle}></div>
      <div className='error-field-wrapper' style={modalStyle}>
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
          <div onClick={handleClose} style={{ cursor: 'pointer' }}>
            <GradationButton text={'OK'} style1={style} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default ErrorModal;
