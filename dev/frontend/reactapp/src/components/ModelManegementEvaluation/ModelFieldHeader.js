import React, { useState } from 'react';
import './ModelManegementEvaluation.css';
import { ReactComponent as DeleteIcon } from '../../assets/svg/delete_24.svg'
import { ReactComponent as TriangleSVG } from '../../assets/svg/arrow_drop_down_24.svg';
import { ReactComponent as RemoveSVG } from '../../assets/svg/remove_24.svg';

function ModelFieldHeader({ accuracySort, lossSort, dateSort, handleDelate }) {
  const [isAccuracyAcending, setIsAccuracyAcending] = useState(true);
  const [isLossAcending, setIsLossAcending] = useState(true);
  const [isDateAcending, setIsDateAcending] = useState(true);
  const [isAccuracy, setIsAccuracy] = useState(false);
  const [isLoss, setIsLoss] = useState(false);
  const [isDate, setIsDate] = useState(false);
  const accuracyStyle = {
    transform: isAccuracyAcending ? 'rotateX(0deg)' : 'rotateX(180deg)'
  }
  const lossStyle = {
    transform: isLossAcending ? 'rotateX(0deg)' : 'rotateX(180deg)'
  }
  const dateStyle = {
    transform: isDateAcending ? 'rotateX(0deg)' : 'rotateX(180deg)'
  }
  const handleAccuracy = () => {
    setIsAccuracyAcending(!isAccuracyAcending);
    accuracySort(isAccuracyAcending);
    setIsAccuracy(true);
    setIsLoss(false);
    setIsDate(false);
  };
  const handleLoss = () => {
    setIsLossAcending(!isLossAcending);
    lossSort(isLossAcending);
    setIsAccuracy(false);
    setIsLoss(true);
    setIsDate(false);
  }
  const handleDate = () => {
    setIsDateAcending(!isDateAcending);
    dateSort(isDateAcending);
    setIsAccuracy(false);
    setIsLoss(false);
    setIsDate(true);
  };
  return (
    <div className='model-field-header-wrapper'>
      <div className='model-name-div'>
        <p>Name</p>
      </div>
      <div className='model-accuracy-div' onClick={handleAccuracy} style={{ cursor: 'pointer' }}>
        {isAccuracy && <TriangleSVG className='model-header-svg' style={accuracyStyle} />}
        {!isAccuracy && <RemoveSVG className='model-header-svg' />}
        <p>Accuracy</p>
      </div>
      <div className='model-loss-div' onClick={handleLoss} style={{ cursor: 'pointer' }}>
        {isLoss && <TriangleSVG className='model-header-svg' style={lossStyle} />}
        {!isLoss && <RemoveSVG className='model-header-svg' />}
        <p>Loss</p>
      </div>
      <div className='model-date-div' onClick={handleDate} style={{ cursor: 'pointer' }}>
        {isDate && <TriangleSVG className='model-header-svg' style={dateStyle} />}
        {!isDate && <RemoveSVG className='model-header-svg' />}
        <p>Date</p>
      </div>
      <div className='model-delet-div'>
        <div className='model-delet-icon-wrapper' onClick={handleDelate} style={{ cursor: 'pointer' }}>
          <DeleteIcon className='model-delet-svg' />
        </div>
      </div>
    </div>
  )
}

export default ModelFieldHeader;
