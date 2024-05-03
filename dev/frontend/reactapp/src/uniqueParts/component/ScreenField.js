import React from 'react';
import '../css/ScreenField.css';
import EditScreen from './EditScreen';

function ScreenField() {
  return (
    <div className='screen-field-wrapper'>
      <div className='left-screen'>
        <EditScreen />
      </div>
      <div className='right-screen'>
        <div className='top-screen'></div>
        <div className='bottom-screen'></div>
      </div>
    </div>
  )
}

export default ScreenField
