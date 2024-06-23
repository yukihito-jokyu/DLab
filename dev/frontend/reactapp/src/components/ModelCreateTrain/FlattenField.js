import React from 'react'
import './ModelCreateTrain.css'
import Tile from './Tile/Tile'

function FlattenField() {
  const text = 'Flatten';
  const style = {
    backgroundColor: '#46C17E'
  }
  return (
    <div className='flatten-field-wrapper'>
      <Tile text={text} style={style} />
    </div>
  )
}

export default FlattenField
