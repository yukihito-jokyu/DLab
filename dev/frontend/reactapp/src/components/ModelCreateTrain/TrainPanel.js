import React from 'react';
import './TrainPanel.css';
import TrainPanelTital from './TrainPanelTital';
import TrainPanelEdit from './TrainPanelEdit';

function TrainPanel() {
  return (
    <div className='train-panel-wrapper'>
      <TrainPanelTital />
      <TrainPanelEdit />
    </div>
  )
}

export default TrainPanel
