import React from 'react';
import './TrainPanel.css';
import TrainPanelTital from './TrainPanelTital';
import TrainPanelEdit from './TrainPanelEdit';

function TrainPanel() {
  return (
    <div className='train-panel-wrapper'>
      <TrainPanelTital />
      <div className='panel-field'>
        <TrainPanelEdit />
        <TrainPanelEdit />
        <TrainPanelEdit />
        <TrainPanelEdit />
        <TrainPanelEdit />
        <TrainPanelEdit />
        <TrainPanelEdit />
        <TrainPanelEdit />
      </div>
    </div>
  )
}

export default TrainPanel
