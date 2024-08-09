import React from 'react'
import TrainPanelTital from '../TrainPanelTital'

function Loss() {
  return (
    <div className='train-log-wrapper'>
      <TrainPanelTital title={'Loss'} />
      <div className='log-field'>
        {/* ここに学習曲線 */}
      </div>
    </div>
  )
}

export default Loss
