import React from 'react'
import './ModelCreateTrain.css'
import Tile from './Tile/Tile'

function FlattenField({ flattenWay, setLayerType, flattenShape }) {
  const text = flattenWay.type;
  const style = {
    backgroundColor: '#46C17E'
  }
  const handleClick = () => {
    setLayerType(flattenWay.type)
  }
  return (
    <div className='flatten-field-wrapper'>
      <div onClick={handleClick}>
        <Tile text={text} shape={flattenShape} style={style} />
      </div>
    </div>
  )
}

export default FlattenField
