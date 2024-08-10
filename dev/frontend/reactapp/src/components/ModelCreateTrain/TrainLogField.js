import React, { useState } from 'react';
import './TrainLogField.css';
import DisplayAcc from '../ModelManegementEvaluation/DisplayAcc';
import DisplayLoss from '../ModelManegementEvaluation/DisplayLoss';
import TrainPanelTital from './TrainPanelTital';
import './log.css';
import { useParams } from 'react-router-dom';
import useFetchTrainingResults from '../../hooks/useFetchStatus';

function TrainLogField() {
  // const [modelId, setModelId] = useState('');
  // setModelId(useParams());
  // const { accuracyData, lossData } = useFetchTrainingResults(modelId);
  return (
    <div className='train-log-field-wrapper'>
      <div className='train-log-wrapper'>
        <TrainPanelTital title={'Accuracy'} />
        <div className='log-field'>
          {/* <DisplayAcc accuracyData={accuracyData} /> */}
        </div>
      </div>
      <div className='train-log-wrapper'>
        <TrainPanelTital title={'Loss'} />
        {/* <DisplayLoss lossData={lossData} /> */}
        <div className='log-field'>
        </div>
      </div>
    </div>
  )
}

export default TrainLogField
