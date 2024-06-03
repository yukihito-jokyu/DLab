import React, { useEffect, useState } from 'react';

function Recommend() {
  const [height, setHeight] = useState(0);

  useEffect(() => {
    const calculateHeight = () => {
      const element = document.getElementById('recommend-icon-width');
      const width = element.offsetWidth;
      console.log(width)
      setHeight(width);
    };

    // 初期計算
    calculateHeight();

    window.addEventListener('resize', calculateHeight);
    return () => window.removeEventListener('resize', calculateHeight);
  }, []);
  const iconStyle = {
    height: `${height}px`
  }
  return (
    <div className='recommend-wrapper'>
      <div className='recommend-comment'>
        <p>こんな方におすすめ</p>
      </div>
      <div className='icon-wrapper'>
        <div className='recommend-icon'id='recommend-icon-width' style={iconStyle}>
          <p>1</p>
        </div>
        <div className='recommend-icon'  style={iconStyle}>
          <p>2</p>
        </div>
        <div className='recommend-icon'  style={iconStyle}>
          <p>3</p>
        </div>
      </div>
    </div>
  );
};

export default Recommend;
