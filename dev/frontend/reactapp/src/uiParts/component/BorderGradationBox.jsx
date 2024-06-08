import React from 'react';
import '../css/BorderGradationBox.css';

function BorderGradationBox({ children, style1, style2}) {
  // const ChildComponent = props.children;
  // const style1 = props.style1;
  // const style2 = props.style2;
  return (
    <div className='gradation-border' style={style1}>
      <div className='gradation-wrapper' style={style2}>
        {children}
      </div>
    </div>
  );
};

export default BorderGradationBox;