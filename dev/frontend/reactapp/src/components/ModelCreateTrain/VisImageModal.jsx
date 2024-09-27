import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import useFetchStatus from '../../hooks/useFetchStatus';
import './ModelCreateTrain.css';
import { ReactComponent as DeletIcon } from '../../assets/svg/delet_48.svg';
import GradationFonts from '../../uiParts/component/GradationFonts';
import { ReactComponent as EastIcon } from '../../assets/svg/east_24.svg';

function VisImageModal({ changeVisImageModal, image, label, preLabel, epoch }) {
  console.log(epoch)
  const { modelId } = useParams();
  const [isVisible, setIsVisible] = useState(false);
  const [opacity, setOpacity] = useState(0.8);
  const isMatch = label === preLabel;
  const currentStatus = useFetchStatus(modelId);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  useEffect(() => {
    if (currentStatus === 'done') {
      changeVisImageModal();
    }
  }, [currentStatus, changeVisImageModal]);

  useEffect(() => {
    setOpacity(0.8);
    const timeout = setTimeout(() => {
      setOpacity(0);
    }, 1000);

    return () => clearTimeout(timeout);
  }, [label, preLabel]);

  const modalStyle = {
    opacity: isVisible ? 1 : 0,
    transition: 'opacity 0.5s ease, transform 0.3s ease',
  };

  const indicatorStyle = {
    fontSize: '500px',
    color: isMatch ? '#FF0000' : '#128DF2',
    opacity: opacity,
    transition: 'opacity 1s ease',
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
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
              {image && <p style={indicatorStyle}>
                {isMatch ? '○' : '✕'}
              </p>}
              <div className='modal-title'>
                <GradationFonts text={"画像分類結果"} style={{ fontSize: "30px", fontWeight: 600 }} />
              </div>
              <div className='gradation-border2-wrapper'>
                <div className='gradation-border2'></div>
              </div>
              <div className='epoch-wrapper'>
                {epoch && <p>{epoch} epoch</p>}
              </div>
              <div className='vis-image-images-wrapper'>
                <div className='image-label-wraper'>
                  <div className='label-wrapper'>
                    <p>{label}</p>
                  </div>
                  {image && <img src={`data:image/png;base64,${image}`} alt='test_image' />}
                </div>
                <div className='svg-wrapper'>
                  {image && <EastIcon className='east-svg' />}
                </div>
                <div className='prelabel-wrapper'>
                  {preLabel && <p>{preLabel}</p>}
                </div>
              </div>
              <div className='vis-image-modal-delet-button-field' onClick={handleClose}>
                <DeletIcon className='vis-delet-svg' />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VisImageModal;
