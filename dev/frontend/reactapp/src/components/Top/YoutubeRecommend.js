import React, { useEffect, useState } from 'react';
import GradationButton from '../../uiParts/component/GradationButton';

function YoutubeRecommend() {
  const [height, setHeight] = useState(0);
  useEffect(() => {
    const calculateHeight = () => {
      const element = document.getElementById('youtube-field-width');
      const width = element.offsetWidth / 3 * 2;
      setHeight(width);
    };

    calculateHeight()
    window.addEventListener('resize', calculateHeight);
    return () => window.removeEventListener('resize', calculateHeight);
  }, [])
  const youtubeStyle = {
    background: '#E37588'
  }
  const youtubeFieldStyle = {
    height: `${height}px`
  }
  return (
    <div className='youtube-wrapper'>
      <div className='youtube-title'>
        <p>HOW TO USE</p>
      </div>
      <div className='youtube-info-wrapper'>
        <div className='youtube-info' style={youtubeFieldStyle}>
          <div className='youtube-middle'>
            <p className='info-p'>下記リンクより利用方法等を<br />確認することができます</p>
            <div className='youtube-button'>
              <GradationButton text={'Youtube'} style1={youtubeStyle} />
            </div>
          </div>
        </div>
        <div className='youtube-field' id='youtube-field-width' style={youtubeFieldStyle}></div>
      </div>
    </div>
  );
};

export default YoutubeRecommend;
