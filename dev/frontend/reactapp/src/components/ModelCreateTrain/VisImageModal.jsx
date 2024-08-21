import React, { useEffect, useState } from 'react'
import './ModelCreateTrain.css'
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';
import GradationFonts from '../../uiParts/component/GradationFonts';

function VisImageModal({ changeVisImageModal }) {

  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const modalStyle = {
    opacity: isVisible ? 1 : 0,
    transition: 'opacity 0.5s ease, transform 0.3s ease',
  };

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      changeVisImageModal();
    }, 200);
  };

  return (
    <div>
      <div className='train-modal-wrapper'></div>
      <div className='tile-add-field-wrapper'>
        <div className='gradation-border' style={modalStyle}>
          <div className='gradation-wrapper'>
            <div className='vis-image-modal-field'>
              <div className='modal-title'>
                <GradationFonts text={"画像分類結果"} style={{ fontSize: "30px", fontWeight: 600 }} />
              </div>
              <div className='gradation-border2-wrapper'>
                <div className='gradation-border2'></div>
              </div>


              ここに表示したいことを記述


              <div className='vis-image-modal-delet-button-field' onClick={handleClose}>
                <DeletIcon className='vis-delet-svg' />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default VisImageModal;
