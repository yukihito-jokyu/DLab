import React from 'react';
import './EditTileParamet.css';

function EditTileParamet({ name, value }) {
  return (
    <div className='edit-tile-paramet-wrapper'>
      <div className='edit-tile-name-wrapper'>
        <p>{name}</p>
      </div>
      <div className='edit-tile-value-wrapper'>
        <p>{value}</p>
      </div>
    </div>
  )
}

export default EditTileParamet
