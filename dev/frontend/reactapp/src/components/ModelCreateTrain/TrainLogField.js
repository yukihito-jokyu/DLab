import React from 'react';
import './TrainLogField.css';
import TrainPanel from './TrainPanel';
import TrainLog from './TrainLog';
import Accuracy from './Log/Accuracy';
import Loss from './Log/Loss';

function TrainLogField({ trainInfo, setTrainInfo }) {
  return (
    <div className='train-log-field-wrapper'>
      <Accuracy />
      <Loss />
    </div>
  )
}

export default TrainLogField
