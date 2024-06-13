import React from 'react';
import './ModelManegementEvaluation.css';
import { ReactComponent as DeleteIcon } from '../../assets/svg/delete_24.svg'

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
        <div className='model-delet-icon-wrapper'>
          <DeleteIcon className='model-delet-svg' />
        </div>
      </div>
    </div>
  )
}

export default ModelFieldHeader;
