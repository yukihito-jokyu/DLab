import React from 'react';
import { useParams } from 'react-router-dom';
import useFetchStatus from '../../hooks/useFetchStatus';
import '../css/TrainButtons.css';
import { ReactComponent as TerminalIcon } from '../../assets/svg/terminal_24.svg'
import { ReactComponent as PlayIcon } from '../../assets/svg/play_circle_24.svg'
import { ReactComponent as ImageIcon } from '../../assets/svg/image.svg';

function TrainButtons({ changeEdit, changeTrain }) {
  const { modelId } = useParams();
  const currentStatus = useFetchStatus(modelId);

  return (
    <div className='train-buttons-wrapper'>
      <div className='image-wrapper' style={{ cursor: 'pointer' }}>
        <ImageIcon className='image-svg' />
      </div>
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
  )
}

export default TrainButtons;
