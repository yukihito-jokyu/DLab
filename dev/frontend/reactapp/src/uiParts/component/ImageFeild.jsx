import React from 'react'

function ImageFeild(props) {
  const elementSize = props.elementsize;
  const style = {
    width: elementSize,
    height: elementSize
  };
  return (
    <div style={style}>
      写真
    </div>
  )
}

export default ImageFeild
