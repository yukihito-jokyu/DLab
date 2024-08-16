import React, { useState } from 'react';
import './TrainLogField.css';
import DisplayResult from '../ModelManegementEvaluation/DisplayResult';
import TrainPanelTital from './TrainPanelTital';
import './log.css';
import { useParams } from 'react-router-dom';
import useFetchTrainingResults from '../../hooks/useFetchTrainingResults';

function TrainLogField() {
  const { modelId } = useParams();
  const { accuracyData, lossData } = useFetchTrainingResults(modelId);
  console.log(`accuracy:${accuracyData}\nloss:${lossData}`);

  return (
    <div className='train-log-field-wrapper'>
      <div className='train-log-wrapper'>
        <TrainPanelTital title={'Accuracy'} />
        <div className='log-field'>
          {accuracyData && accuracyData.labels ? (
            <DisplayResult data={accuracyData} type="Accuracy" showTitle={false} />
          ) : (
            <></>
          )}
        </div>
      </div>
      <div className='train-log-wrapper'>
        <TrainPanelTital title={'Loss'} />
        <div className='log-field'>
          {lossData && lossData.labels ? (
            <DisplayResult data={lossData} type="Loss" showTitle={false} />
          ) : (
            <></>
          )}
        </div>
      </div>
    </div>
  );
}

export default TrainLogField;
