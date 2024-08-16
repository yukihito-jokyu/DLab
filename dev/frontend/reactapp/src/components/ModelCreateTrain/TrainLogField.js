import React from 'react';
import './TrainLogField.css';
import DisplayResult from '../ModelManegementEvaluation/DisplayResult';
import TrainPanelTital from './TrainPanelTital';
import './log.css';
import { useParams } from 'react-router-dom';
import useFetchTrainingResults from '../../hooks/useFetchTrainingResults';

function TrainLogField() {
  const { modelId } = useParams();
  const { currentTask, accuracyData, lossData, totalRewardData, averageLossData } = useFetchTrainingResults(modelId);

  return (
    <div className='train-log-field-wrapper'>
      {currentTask === 'ImageClassification' && (
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
      {currentTask === 'ReinforcementLearning' && (
        <>
          <div className='train-log-wrapper'>
            <TrainPanelTital title={'Total Reward'} />
            <div className='log-field'>
              {totalRewardData && totalRewardData.labels ? (
                <DisplayResult data={totalRewardData} type="Total Reward" showTitle={false} />
              ) : (
                <></>
              )}
            </div>
          </div>
          <div className='train-log-wrapper'>
            <TrainPanelTital title={'Average Loss'} />
            <div className='log-field'>
              {averageLossData && averageLossData.labels ? (
                <DisplayResult data={averageLossData} type="Average Loss" showTitle={false} />
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
