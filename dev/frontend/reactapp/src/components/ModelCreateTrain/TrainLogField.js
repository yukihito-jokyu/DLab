import React from 'react';
import './TrainLogField.css';
import DisplayResult from '../ModelManegementEvaluation/DisplayResult';
import TrainPanelTital from './TrainPanelTital';
import './log.css';
import { useParams } from 'react-router-dom';
import useFetchTrainingResults from '../../hooks/useFetchTrainingResults';

function TrainLogField() {
  const { modelId, task } = useParams();
  const { accuracyData, lossData, totalRewardData, averageLossData } = useFetchTrainingResults(modelId, task);

  return (
    <div className='train-log-field-wrapper'>
      {task === 'ImageClassification' && (
        <>
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
        </>
      )}
      {task === 'ReinforcementLearning' && (
        <>
          <div className='train-log-wrapper'>
            <TrainPanelTital title={'Reward'} />
            <div className='log-field'>
              {totalRewardData && totalRewardData.labels ? (
                <DisplayResult data={totalRewardData} type="Reward" showTitle={false} />
              ) : (
                <></>
              )}
            </div>
          </div>
          <div className='train-log-wrapper'>
            <TrainPanelTital title={'Loss'} />
            <div className='log-field'>
              {averageLossData && averageLossData.labels ? (
                <DisplayResult data={averageLossData} type="Loss" showTitle={false} />
              ) : (
                <></>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default TrainLogField;
