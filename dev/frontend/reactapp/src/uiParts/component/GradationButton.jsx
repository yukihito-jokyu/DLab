import React from 'react';
import '../css/GradationButton.css';

function GradationButton(props) {
  const text = props.text;
  const style1 = props.style1;
  const style2 = props.style2;
  return (
    <div className='gradation-button-wrapper' style={style1}>
      <p style={style2}>{text}</p>
    </div>
  )
}

export default GradationButton
