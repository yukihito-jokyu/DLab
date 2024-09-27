import React, { useEffect, useState } from 'react';
import useFetchStatus from '../../hooks/useFetchStatus';
import { useParams } from 'react-router-dom';
import './ModelCreateTrain.css'
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';
import GradationFonts from '../../uiParts/component/GradationFonts';

function VisTrainModal({ changeVisTrainModal, image, epoch }) {
  const { modelId } = useParams();
  const [isVisible, setIsVisible] = useState(false);
  const currentStatus = useFetchStatus(modelId);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  useEffect(() => {
    if (currentStatus === 'done') {
      changeVisTrainModal();
    }
  }, [currentStatus, changeVisTrainModal]);

  const modalStyle = {
    opacity: isVisible ? 1 : 0,
    transition: 'opacity 0.5s ease, transform 0.3s ease',
  };

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      changeVisTrainModal();
    }, 200);
  };



  return (
    <div>
      <div className='train-modal-wrapper'></div>
      <div className='tile-add-field-wrapper'>
        <div className='gradation-border' style={modalStyle}>
          <div className='gradation-wrapper'>
            <div className='vis-train-modal-field'>
              <div className='modal-title'>
                <GradationFonts text={"学習中描画"} style={{ fontSize: "30px", fontWeight: 600 }} />
              </div>
              <div className='gradation-border2-wrapper'>
                <div className='gradation-border2'></div>
              </div>
              <div className='epoch-wrapper'>
                {epoch && <p>{epoch} epoch</p>}
              </div>
              <div className='vis-train-images-wrapper'>
                {image && <img src={`data:image/png;base64,${image}`} alt='test_image' />}
              </div>
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

export default VisTrainModal;
