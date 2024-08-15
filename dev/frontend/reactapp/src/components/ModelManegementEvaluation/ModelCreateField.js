import React, { useEffect, useState } from 'react';
import ModelCreateBox from './ModelCreateBox';
import BorderGradationBox from '../../uiParts/component/BorderGradationBox';

function ModelCreateField({ handleCreateModal, setSuccessModelCreate, setSameModelName }) {
  const [isVisible, setIsVisible] = useState(false);
  const [isRendered, setIsRendered] = useState(true);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      setIsRendered(false);
      handleCreateModal();
    }, 200);
  };

  const style1 = {
    width: '906px',
    height: '306px'
  };

  const style2 = {
    position: 'relative',
    padding: '0px'
  };

  const backgroundWrapperStyle = {
    opacity: isVisible ? 1 : 0,
    transform: isVisible ? 'scale(1)' : 'scale(0.8)',
    transition: 'opacity 0.2s ease, transform 0.2s ease',
  };

  const backgroundColorStyle = {
    opacity: isVisible ? 0.3 : 0,
  };

  return (
    isRendered && (
      <div className='create-background-wrapper' style={backgroundWrapperStyle}>
        <div className='create-background-color' style={backgroundColorStyle}></div>
        <BorderGradationBox style1={style1} style2={style2} >
          <ModelCreateBox handleCreateModal={handleClose} setSuccessModelCreate={setSuccessModelCreate} setSameModelName={setSameModelName} />
        </BorderGradationBox>
      </div>
    )
  );
}

export default ModelCreateField;
