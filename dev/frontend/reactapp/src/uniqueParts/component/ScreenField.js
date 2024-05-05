import React from 'react';
import '../css/ScreenField.css';
import EditScreen from './EditScreen';
import DataScreen from './DataScreen';
import EditTileParameterField from './EditTileParameterField';

function ScreenField() {
  return (
    <div className='screen-field-wrapper'>
      <div className='left-screen'>
        <EditScreen />
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
