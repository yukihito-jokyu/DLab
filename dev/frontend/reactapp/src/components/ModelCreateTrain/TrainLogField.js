import React from 'react';
import './TrainLogField.css';
import TrainPanel from './TrainPanel';
import TrainLog from './TrainLog';

function TrainLogField() {
  return (
    <div className='train-log-field-wrapper'>
      <TrainPanel />
      <TrainLog />
    </div>
  )
}

export default TrainLogField
