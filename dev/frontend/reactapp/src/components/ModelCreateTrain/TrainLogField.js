import React from 'react';
import './TrainLogField.css';
import TrainPanel from './TrainPanel';
import TrainLog from './TrainLog';

function TrainLogField({ trainInfo, setTrainInfo }) {
  return (
    <div className='train-log-field-wrapper'>
      <TrainPanel
        trainInfo={trainInfo}
        setTrainInfo={setTrainInfo}
      />
      <TrainLog />
    </div>
  )
}

export default TrainLogField
