import React from 'react'
import TrainPanelTital from '../TrainPanelTital'
import './log.css';

function Accuracy() {
  return (
    <div className='train-log-wrapper'>
      <TrainPanelTital title={'Accuracy'} />
      <div className='log-field'>
        {/* ここに学習曲線 */}
      </div>
    </div>
  )
}

export default Accuracy
