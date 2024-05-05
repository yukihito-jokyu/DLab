import React from 'react';
import '../css/TrainPanel.css';
import TrainPanelTital from '../../uiParts/component/TrainPanelTital';
import TrainPanelEdit from '../../uiParts/component/TrainPanelEdit';

function TrainPanel() {
  return (
    <div className='train-panel-wrapper'>
      <TrainPanelTital />
      <TrainPanelEdit />
    </div>
  )
}

export default TrainPanel
