import React, { useEffect, useState } from 'react';
import './NuronTile.css';

function NuronTile({ shape }) {
  const [shapeText, setShapeText] = useState('');
  const [errorTile, setErrorTile] = useState(false);
  useEffect(() => {
    const initShape = () => {
      setErrorTile(false);
      if (shape) {
        let text = ''
      shape.forEach((num, index) => {
        if (index > 0) {
          text += ','
        }
        text += num;
        if (num < 1 || num === 'error') {
          setErrorTile(true);
        }
      })
      setShapeText(text);
      }
    };
    initShape();
  }, [shape])
  return (
    <div className='nuron-tile-wrapper'>
      {errorTile && <div className='error-wrapper'></div>}
      <div className='nuron-tile'>
        <div className='tile-title-wrapper'>
          <p className='tile-title'>Linear</p>
        </div>
        <div className='output-dim-wrapper'>
          <p>{shapeText}</p>
        </div>
      </div>
      <div className='activate-tile-wrapper'>
        <div className='activate-tile'>
          <p className='activate'>ReLU</p>
        </div>
      </div>
    </div>
  )
}

export default NuronTile
