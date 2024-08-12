import React, { useState } from 'react';
import './TrainLogField.css';
import DisplayAcc from '../ModelManegementEvaluation/DisplayAcc';
import DisplayLoss from '../ModelManegementEvaluation/DisplayLoss';
import TrainPanelTital from './TrainPanelTital';
import './log.css';
import { useParams } from 'react-router-dom';
import useFetchTrainingResults from '../../hooks/useFetchTrainingResults';

function TrainLogField() {

  const { modelId } = useParams()
  const { accuracyData, lossData } = useFetchTrainingResults(modelId);

  return (
    <div className='train-log-field-wrapper'>
      <div className='train-log-wrapper'>
        <TrainPanelTital title={'Accuracy'} />
        <div className='log-field'>
          {accuracyData !== null ? (
            <DisplayAcc accuracyData={accuracyData} showTitle={false} />
          ) : (
            <></>
          )}
        </div>
      </div>
      <div className='train-log-wrapper'>
        <TrainPanelTital title={'Loss'} />
        <div className='log-field'>
          {lossData !== null ? (
            <DisplayLoss lossData={lossData} showTitle={false} />
          ) : (
            <></>
          )}
        </div>
      </div>
    </div>
  )
}

export default TrainLogField
