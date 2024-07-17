import React from 'react';
import '../css/TrainButtons.css';
import { ReactComponent as TerminalIcon } from '../../assets/svg/terminal_24.svg'
import { ReactComponent as PlayIcon } from '../../assets/svg/play_circle_24.svg'

function TrainButtons({ changeEdit, changeTrain }) {
  return (
    <div className='train-buttons-wrapper'>
      <div className='terminal-wrapper' onClick={changeEdit}>
        <TerminalIcon className='terminal-svg' />
      </div>
      <div className='play-wrapper' onClick={changeTrain}>
        <PlayIcon className='play-svg' />
      </div>
    </div>
  )
}

export default TrainButtons
