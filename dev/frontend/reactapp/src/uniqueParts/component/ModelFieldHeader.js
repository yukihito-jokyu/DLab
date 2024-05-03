import React from 'react';
import '../css/ModelFieldHeader.css';
import ModelDeletIcon from '../../uiParts/component/ModelDeletIcon';

function ModelFieldHeader() {
  return (
    <div className='model-field-header-wrapper'>
      <div className='model-name-div'>
        <p>Name</p>
      </div>
      <div className='model-accuracy-div'>
        <p>Accuracy</p>
      </div>
      <div className='model-loss-div'>
        <p>Loss</p>
      </div>
      <div className='model-date-div'>
        <p>Date</p>
      </div>
      <div className='model-delet-div'>
        <ModelDeletIcon />
      </div>
    </div>
  )
}

export default ModelFieldHeader;
