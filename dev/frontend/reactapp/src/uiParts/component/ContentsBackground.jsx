import React from 'react';
import '../css/ContentsBackground.css';

function ContentsBackground(props) {
  const ChildComponent = props.children;
  const style = props.style;
  return (
    <div className='contents-background-style' style={style}>
      <ChildComponent />
    </div>
  );
};

export default ContentsBackground;