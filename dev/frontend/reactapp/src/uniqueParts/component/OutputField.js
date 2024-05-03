import React from 'react';
import '../css/OutputField.css';
import OutputTile from '../../uiParts/component/OutputTile';

function OutputField() {
  return (
    <div className='output-field-wrapper'>
      <div className='output-tile-position'>
        <OutputTile />
      </div>
    </div>
  )
}

export default OutputField
