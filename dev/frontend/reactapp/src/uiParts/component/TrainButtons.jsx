import React from 'react';
import { useParams } from 'react-router-dom';
import useFetchStatus from '../../hooks/useFetchStatus';
import '../css/TrainButtons.css';
import { ReactComponent as TerminalIcon } from '../../assets/svg/terminal_24.svg';
import { ReactComponent as PlayIcon } from '../../assets/svg/play_circle_24.svg';
import { ReactComponent as ImageIcon } from '../../assets/svg/image.svg';
import { ReactComponent as RobotIcon } from '../../assets/svg/robot.svg';

const STATUS_CONFIG = {
  pre: { text: '学習前', style: { backgroundColor: '#50A5FF' }, right: '790px' },
  doing: { text: '学習中', style: { backgroundColor: '#8877EB' }, right: '790px' },
  done: { text: '学習済み', style: { backgroundColor: '#46C17E' }, right: '810px' },
};

function TrainButtons({ changeEdit, changeTrain, changeVisImageModal, changeVisTrainModal }) {
  const { modelId, task } = useParams();
  const currentStatus = useFetchStatus(modelId);
  const config = STATUS_CONFIG[currentStatus] || {};

  return (
    <div
      className='train-buttons-wrapper'
      style={{ right: config.right }}
    >
      {task === 'ImageClassification' && (
        <div className='status-and-icon-wrapper'>
          <div className='status-text' style={config.style}>
            {config.text}
          </div>
          <div
            className='image-wrapper'
            onClick={currentStatus === 'doing' ? changeVisImageModal : null}
            style={{
              cursor: currentStatus === 'doing' ? 'pointer' : 'not-allowed',
              opacity: currentStatus === 'doing' ? 1 : 0.7,
            }}
          >
            <ImageIcon className='image-svg' />
          </div>
        </div>
      )}
      {task === 'ReinforcementLearning' && (
        <div className='status-and-icon-wrapper'>
          <div className='status-text' style={config.style}>
            {config.text}
          </div>
          <div
            className='robot-wrapper'
            onClick={currentStatus === 'doing' ? changeVisTrainModal : null}
            style={{
              cursor: currentStatus === 'doing' ? 'pointer' : 'not-allowed',
              opacity: currentStatus === 'doing' ? 1 : 0.7,
            }}
          >
            <RobotIcon className='robot-svg' />
          </div>
        </div>
      )}
      <div className='terminal-wrapper' onClick={changeEdit} style={{ cursor: 'pointer' }}>
        <TerminalIcon className='terminal-svg' />
      </div>
      <div
        className='play-wrapper'
        onClick={currentStatus !== 'doing' ? changeTrain : null}
        style={{
          cursor: currentStatus !== 'doing' ? 'pointer' : 'not-allowed',
          opacity: currentStatus !== 'doing' ? 1 : 0.7,
        }}
      >
        <PlayIcon className='play-svg' />
      </div>
    </div>
  );
}

export default TrainButtons;
