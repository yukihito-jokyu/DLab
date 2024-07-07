import React, { useEffect, useState } from 'react'
import './Tile.css'

function Tile({ text, shape }) {
  const [individualStyle, setIndividualStyle] = useState(null);
  const [shapeText, setShapeText] = useState('');
  const [errorTile, setErrorTile] = useState(false);
  useEffect(() => {
    const initShape = () => {
      setErrorTile(false);
      if (shape) {
        let text = ''
      console.log(shape)
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
  useEffect(() => {
    const initStyle = () => {
      if (text === 'MaxPool2d') {
        const style = {
          backgroundColor: "#D36B5D"
        }
        setIndividualStyle(style);
      } else if (text === 'Flatten') {
        const style = {
          backgroundColor: "#46C17E"
        }
        setIndividualStyle(style);
      } else if (text === 'Conv2d') {
        const style = {
          backgroundColor: "#8E97EA"
        }
        setIndividualStyle(style);
      } else if (text === 'BatchNorm') {
        const style = {
          backgroundColor: "#62BFDC"
        }
        setIndividualStyle(style);
      } else if (text === 'Dropout') {
        const style = {
          backgroundColor: "#AC6ACB"
        }
        setIndividualStyle(style);
      }
      
    };

    initStyle();
  }, [text]);
  return (
    <div className='tile-wrapper' style={individualStyle}>
      {errorTile && <div className='error-wrapper'></div>}
      <div className='tile-title-wrapper'>
        <p className='tile-title'>{text}</p>
      </div>
      <div className='output-dim-wrapper'>
        <p>{shapeText}</p>
      </div>
    </div>
  )
}

export default Tile
