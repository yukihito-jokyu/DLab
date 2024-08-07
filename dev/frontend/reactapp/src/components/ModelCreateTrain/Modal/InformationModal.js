import React from 'react';
import './InformationModal.css';
import GradationFonts from '../../../uiParts/component/GradationFonts';
import { ReactComponent as DeletIcon } from '../../../assets/svg/delet_40.svg';

function InformationModal({ infoName, handleDelete }) {
  // const text = 'カーネルサイズ'
  const fontStyle1 = {
    fontSize: '26px',
    fontWeight: '600'
  }
  return (
    <div>
      <div className='information-modal-wrapper'></div>
      <div className='tile-add-field-wrapper'>
        <div className='gradation-border'>
          <div className='gradation-wrapper'>
            <div className='info-modal-field'>
              <div className='modal-title'>
                <GradationFonts text={infoName} style={fontStyle1} />
              </div>
              <div className='gradation-border2-wrapper'>
                <div className='gradation-border2'></div>
              </div>
              <div className='exp-field'>
                <p>ああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああああ</p>
              </div>
              <div className='train-modal-delet-button-field' onClick={() => handleDelete(false)}>
                <DeletIcon className='delet-svg' />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default InformationModal;
