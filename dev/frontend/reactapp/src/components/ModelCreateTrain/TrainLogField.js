import React, { useState } from 'react';
import './TrainLogField.css';
import DisplayAcc from '../ModelManegementEvaluation/DisplayAcc';
import DisplayLoss from '../ModelManegementEvaluation/DisplayLoss';
import TrainPanelTital from './TrainPanelTital';
import './log.css';
import { useParams } from 'react-router-dom';
import useFetchTrainingResults from '../../hooks/useFetchTrainingResults';

function TrainLogField() {
  // const [modelId, setModelId] = useState('');
  const { modelId } = useParams()
  // setModelId();
  const { accuracyData, lossData } = useFetchTrainingResults(modelId);
  console.log(accuracyData)
  if (accuracyData !== null) {
    console.log(accuracyData.labels)
  }
  // console.log(accuracyData.labels)
  return (
    <div className='train-log-field-wrapper'>
      <div className='train-log-wrapper'>
        <TrainPanelTital title={'Accuracy'} />
        <div className='log-field'>
          {accuracyData !== null ? (
            <DisplayAcc accuracyData={accuracyData} />
          ) : (
            <></>
          )}
        </div>
      </div>
      <div className='train-log-wrapper'>
        <TrainPanelTital title={'Loss'} />
          {lossData !== null ? (
              <DisplayLoss lossData={lossData} />
            ) : (
              <></>
            )}
        <div className='log-field'>
        </div>
      </div>
    </div>
  )
}

export default TrainLogField
