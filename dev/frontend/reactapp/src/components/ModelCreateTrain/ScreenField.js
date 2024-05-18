import React from 'react';
import './ScreenField.css';
import EditScreen from './EditScreen';
import DataScreen from './DataScreen';
import EditTileParameterField from './EditTileParameterField';
import TrainLogField from './TrainLogField';

function ScreenField() {
  return (
    <div className='screen-field-wrapper'>
      <div className='left-screen'>
        <EditScreen />
        <TrainLogField />
      </div>
      <div className='right-screen'>
        <div className='top-screen'>
          <DataScreen />
        </div>
        <div className='bottom-screen'>
          <EditTileParameterField />
        </div>
      </div>
    </div>
  )
}

export default ScreenField
