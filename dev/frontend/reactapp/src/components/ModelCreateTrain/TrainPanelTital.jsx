import React from 'react';
import './TrainPanelTital.css';
import GradationFonts from '../../uiParts/component/GradationFonts';

function TrainPanelTital({ title }) {
  // const text = '学習パネル';
  const style = {
    fontSize: '40px',
    fontWeight: '600',
    marginLeft: '20px'
  }
  return (
    <div className='train-panel-tital-border'>
      {/* <p>学習パネル</p> */}
      <GradationFonts text={title} style={style} />
      {/* <div className='train-panel-tital-wrapper'></div> */}
    </div>
  )
}

export default TrainPanelTital
