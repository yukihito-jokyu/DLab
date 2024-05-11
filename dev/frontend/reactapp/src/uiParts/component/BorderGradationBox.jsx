import React from 'react';
import '../css/BorderGradationBox.css';

function BorderGradationBox(props) {
  const ChildComponent = props.children;
  const style1 = props.style1;
  const style2 = props.style2;
  return (
    <div className='gradation-border' style={style1}>
      <div className='gradation-wrapper' style={style2}>
        <ChildComponent />
      </div>
    </div>
  );
};

export default BorderGradationBox;