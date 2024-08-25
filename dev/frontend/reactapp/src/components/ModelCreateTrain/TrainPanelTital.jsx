import React from 'react';
import './TrainPanelTital.css';
import GradationFonts from '../../uiParts/component/GradationFonts';

function TrainPanelTital({ title }) {
  const style = {
    fontSize: '40px',
    fontWeight: '600',
    marginLeft: '20px'
  }
  return (
    <div className='train-panel-tital-border'>
      <GradationFonts text={title} style={style} />
    </div>
  )
}

export default TrainPanelTital
