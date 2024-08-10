import React, { useState } from 'react';
import './ModelManegementEvaluation.css';
import { ReactComponent as DeleteIcon } from '../../assets/svg/delete_24.svg'

function ModelFieldHeader({ accuracySort, lossSort, dateSort, handleDelate }) {
  const [isAccuracyAcending, setIsAccuracyAcending] = useState(true);
  const [isLossAcending, setIsLossAcending] = useState(true);
  const [isDateAcending, setIsDateAcending] = useState(true);
  const handleAccuracy = () => {
    setIsAccuracyAcending(!isAccuracyAcending);
    accuracySort(isAccuracyAcending);
  };
  const handleLoss = () => {
    setIsLossAcending(!isLossAcending);
    lossSort(isLossAcending);
  }
  const handleDate = () => {
    setIsDateAcending(!isDateAcending);
    dateSort(isDateAcending);
  };
  return (
    <div className='model-field-header-wrapper'>
      <div className='model-name-div'>
        <p>Name</p>
      </div>
      <div className='model-accuracy-div' onClick={handleAccuracy} style={{ cursor: 'pointer' }}>
        <p>Accuracy</p>
      </div>
      <div className='model-loss-div' onClick={handleLoss} style={{ cursor: 'pointer' }}>
        <p>Loss</p>
      </div>
      <div className='model-date-div' onClick={handleDate} style={{ cursor: 'pointer' }}>
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
