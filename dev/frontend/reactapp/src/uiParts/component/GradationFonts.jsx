import React from 'react';
import '../css/GradationFonts.css';

function GradationFonts(props) {
  const text = props.text;
  const style = props.style;
  return (
    <div className='gradation-fonts-wrapper'>
      <p style={style}>{text}</p>
    </div>
  );
};

export default GradationFonts;