import React, { useState } from 'react';
import './ModelCreateTrain.css';
import EditScreen from './EditScreen';
import DataScreen from './DataScreen';
import EditTileParameterField from './EditTileParameterField';
import TrainLogField from './TrainLogField';

function ScreenField() {
  const [parameter, setParameter] = useState(null);
  return (
    <div className='screen-field-wrapper'>
      <div className='left-screen'>
        <EditScreen setParameter={setParameter} />
        {/* <TrainLogField /> */}
      </div>
      <div className='right-screen'>
        <div className='top-screen'>
          <DataScreen />
        </div>
        <div className='bottom-screen'>
          <EditTileParameterField parameter={parameter} />
        </div>
      </div>
    </div>
  )
}

export default ScreenField
